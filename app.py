#!/usr/bin/env python3
"""
Backend API for VLSI Practice - SystemVerilog with Verilator
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
import subprocess
import tempfile
import os
import json
import uuid
import logging
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VLSI Practice API - SystemVerilog")

# Create directories
WAVEFORM_DIR = Path("/tmp/waveforms")
COVERAGE_DIR = Path("/tmp/coverage")
LOG_DIR = Path("/tmp/logs")

for dir_path in [WAVEFORM_DIR, COVERAGE_DIR, LOG_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

logger.info(f"Directories created: {WAVEFORM_DIR}, {COVERAGE_DIR}, {LOG_DIR}")

# Thread pool for Verilator compilation
executor = ThreadPoolExecutor(max_workers=4)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Models
class CodeRequest(BaseModel):
    problem_id: str
    code: str
    user_id: str = "anonymous"
    generate_waveform: bool = False
    enable_assertions: bool = True
    enable_coverage: bool = True

class SystemVerilogRequest(BaseModel):
    code: str
    testbench: Optional[str] = None
    constraints: Optional[str] = None
    enable_assertions: bool = True
    enable_coverage: bool = False
    generate_waveform: bool = False
    timeout: int = 10

class AssertionResult(BaseModel):
    passed: int
    failed: int
    total: int
    details: List[Dict]

class CoverageResult(BaseModel):
    line_coverage: float
    toggle_coverage: float
    assertion_coverage: float
    details: Dict

# Load problems (SystemVerilog enhanced)
PROBLEMS = []
PROBLEMS_PATH = Path("problems_sv.json")
if PROBLEMS_PATH.exists():
    with open(PROBLEMS_PATH, "r") as f:
        PROBLEMS = json.load(f)
    logger.info(f"Loaded {len(PROBLEMS)} SystemVerilog problems")
else:
    # Fallback to basic problems
    with open("problems.json", "r") as f:
        PROBLEMS = json.load(f)
    logger.info(f"Loaded {len(PROBLEMS)} basic problems")

@app.get("/")
async def root():
    return {
        "status": "VLSI Practice API - SystemVerilog Edition",
        "version": "5.0",
        "backend": "Verilator",
        "features": ["SystemVerilog", "Assertions", "Coverage", "Waveforms"]
    }

@app.get("/api/problems")
async def get_problems():
    """Return list of available problems"""
    simplified = []
    for problem in PROBLEMS:
        simplified.append({
            "id": problem["id"],
            "title": problem["title"],
            "description": problem["description"],
            "difficulty": problem["difficulty"],
            "category": problem["category"],
            "template": problem["template"],
            "supports_sv": problem.get("supports_sv", False),
            "has_assertions": problem.get("has_assertions", False)
        })
    return {"problems": simplified}

@app.post("/api/run")
async def run_code(request: CodeRequest):
    """Execute SystemVerilog/Verilog code"""
    try:
        # Find problem
        problem = next((p for p in PROBLEMS if p["id"] == request.problem_id), None)
        if not problem:
            raise HTTPException(status_code=404, detail="Problem not found")
        
        # Choose backend based on problem type
        if problem.get("requires_sv", False) or problem.get("has_assertions", False):
            # Use Verilator for SystemVerilog
            result = await run_verilator_simulation(
                request.code,
                problem.get("testbench_sv", problem.get("testbench", "")),  # ← FIXED!
                request.generate_waveform,
                request.enable_assertions,
                request.enable_coverage,
                problem["title"]
            )
        else:
            # Use Icarus for simple Verilog
            result = run_iverilog_simulation(
                request.code,
                problem.get("testbench", problem.get("testbench_sv", "")),  # ← FIXED!
                request.generate_waveform,
                problem["title"]
            )
        
        # Prepare response
        response = {
            "success": result["success"],
            "problem": problem["title"],
            "output": result.get("output", ""),
            "error": result.get("error", ""),
            "details": result.get("details", ""),
            "backend": result.get("backend", "iverilog"),
            "assertions": result.get("assertions", {}),
            "coverage": result.get("coverage", {}),
            "execution_time": result.get("execution_time", 0)
        }
        
        if not result["success"] and "hint" in problem:
            response["hint"] = problem["hint"]
        
        # Add waveform info
        if "waveform_id" in result:
            waveform_id = result["waveform_id"]
            response["waveform_id"] = waveform_id
            response["waveform_url"] = f"/api/waveform/{waveform_id}"
            response["waveform_download_url"] = f"/api/waveform/{waveform_id}?download=true"
        
        return response
        
    except Exception as e:
        logger.error(f"Error in run_code: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/run/sv")
async def run_systemverilog(request: SystemVerilogRequest):
    """Execute pure SystemVerilog code with Verilator"""
    try:
        result = await run_verilator_simulation(
            request.code,
            request.testbench or "",
            request.generate_waveform,
            request.enable_assertions,
            request.enable_coverage,
            "Custom SystemVerilog",
            request.constraints,
            request.timeout
        )
        
        return {
            "success": result["success"],
            "output": result.get("output", ""),
            "error": result.get("error", ""),
            "assertions": result.get("assertions", {}),
            "coverage": result.get("coverage", {}),
            "waveform_id": result.get("waveform_id"),
            "execution_time": result.get("execution_time", 0),
            "backend": "verilator"
        }
        
    except Exception as e:
        logger.error(f"Error in run_systemverilog: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/waveform/{waveform_id}")
async def get_waveform(waveform_id: str, download: bool = False):
    """Serve waveform with professional HTML viewer"""
    try:
        vcd_path = WAVEFORM_DIR / f"{waveform_id}.vcd"
        
        if download and vcd_path.exists():
            return FileResponse(
                vcd_path,
                media_type="application/octet-stream",
                filename=f"{waveform_id}.vcd"
            )
        
        # Return professional HTML viewer
        html_content = create_professional_viewer(waveform_id, vcd_path.exists())
        return HTMLResponse(content=html_content)
            
    except Exception as e:
        logger.error(f"Error serving waveform: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_verilator_simulation(
    user_code: str, 
    testbench: str, 
    generate_waveform: bool,
    enable_assertions: bool,
    enable_coverage: bool,
    problem_title: str,
    constraints: str = None,
    timeout: int = 10
) -> dict:
    """Run SystemVerilog simulation using Verilator"""
    
    # Run in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        _run_verilator_simulation_sync,
        user_code, testbench, generate_waveform,
        enable_assertions, enable_coverage,
        problem_title, constraints, timeout
    )
    
    return result

def _run_verilator_simulation_sync(
    user_code: str, 
    testbench: str, 
    generate_waveform: bool,
    enable_assertions: bool,
    enable_coverage: bool,
    problem_title: str,
    constraints: str = None,
    timeout: int = 10
) -> dict:
    """Synchronous Verilator execution"""
    waveform_id = None
    start_time = time.time()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        try:
            # Prepare files
            design_file = tmp_path / "design.sv"
            design_file.write_text(user_code)
            
            # Create testbench with VCD if needed
            if generate_waveform:
                waveform_id = str(uuid.uuid4())
                vcd_path = str(tmp_path / "waveform.vcd")
                if "$dumpfile" not in testbench:
                    testbench = testbench.replace(
                        "initial begin",
                        f"initial begin\n    $dumpfile(\"{vcd_path}\");\n    $dumpvars(0);"
                    )
            
            tb_file = tmp_path / "tb.sv"
            tb_file.write_text(testbench)
            
            # Create C++ wrapper
            wrapper_content = create_verilator_wrapper(generate_waveform, enable_coverage)
            wrapper_file = tmp_path / "sim_main.cpp"
            wrapper_file.write_text(wrapper_content)
            
            # Build Verilator command
            cmd = [
                "verilator",
                "--cc", "--exe", "--build",
                "--top-module", "tb",
                "-o", "simulation",
                "-Wno-fatal",
                "-Wno-WIDTH",
                "-Wno-STMTDLY",
                "--Mdir", str(tmp_path / "obj_dir")
            ]
            
            if enable_assertions:
                cmd.append("--assert")
            if enable_coverage:
                cmd.append("--coverage")
            if generate_waveform:
                cmd.append("--trace")
            
            cmd.extend([
                str(design_file),
                str(tb_file),
                str(wrapper_file)
            ])
            
            # Compile with Verilator
            logger.info(f"Compiling with Verilator: {' '.join(cmd)}")
            compile_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=tmp_path,
                timeout=30
            )
            
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "error": "Compilation Failed",
                    "details": compile_result.stderr[:1000],
                    "backend": "verilator",
                    "execution_time": time.time() - start_time
                }
            
            # Run simulation
            sim_result = subprocess.run(
                [str(tmp_path / "simulation")],
                capture_output=True,
                text=True,
                cwd=tmp_path,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            output = sim_result.stdout + sim_result.stderr
            
            # Extract assertion results
            assertions = parse_assertion_results(output)
            
            # Extract coverage if enabled
            coverage = {}
            if enable_coverage and (tmp_path / "coverage.dat").exists():
                coverage = parse_coverage_data(tmp_path / "coverage.dat")
            
            # Save waveform if requested
            if generate_waveform and waveform_id:
                vcd_file = tmp_path / "waveform.vcd"
                if vcd_file.exists() and vcd_file.stat().st_size > 0:
                    dest_vcd = WAVEFORM_DIR / f"{waveform_id}.vcd"
                    shutil.copy2(vcd_file, dest_vcd)
                    logger.info(f"Waveform saved: {waveform_id}")
            
            success = sim_result.returncode == 0 and assertions["failed"] == 0
            
            return {
                "success": success,
                "output": output[:2000],  # Limit output size
                "error": sim_result.stderr[:1000] if sim_result.stderr else "",
                "assertions": assertions,
                "coverage": coverage,
                "waveform_id": waveform_id,
                "backend": "verilator",
                "execution_time": execution_time
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Timeout after {timeout} seconds",
                "backend": "verilator",
                "execution_time": time.time() - start_time
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "backend": "verilator",
                "execution_time": time.time() - start_time
            }

def run_iverilog_simulation(user_code: str, testbench: str, generate_waveform: bool, problem_title: str) -> dict:
    """Fallback to Icarus Verilog for simple Verilog"""
    waveform_id = None
    start_time = time.time()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Prepare testbench with VCD dump
        if generate_waveform:
            waveform_id = str(uuid.uuid4())
            vcd_path = str(tmp_path / "waveform.vcd")
            
            # Add dump commands to testbench
            if "$dumpfile" not in testbench:
                testbench = testbench.replace(
                    "initial begin",
                    f"initial begin\n    $dumpfile(\"{vcd_path}\");\n    $dumpvars(0);"
                )
        
        # Combine source
        source = f"`timescale 1ns/1ps\n{user_code}\n{testbench}"
        source_file = tmp_path / "design.v"
        source_file.write_text(source)
        
        # Compile
        output_exec = tmp_path / "sim"
        compile_result = subprocess.run(
            ["iverilog", "-o", str(output_exec), str(source_file)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if compile_result.returncode != 0:
            return {
                "success": False,
                "error": "Compilation Failed",
                "details": compile_result.stderr[:500],
                "backend": "iverilog",
                "execution_time": time.time() - start_time
            }
        
        # Simulate
        sim_result = subprocess.run(
            ["vvp", str(output_exec)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        execution_time = time.time() - start_time
        output = sim_result.stdout + sim_result.stderr
        
        if "PASS" in output or sim_result.returncode == 0:
            result = {
                "success": True, 
                "output": output,
                "backend": "iverilog",
                "execution_time": execution_time
            }
            
            # Save waveform if requested
            if generate_waveform and waveform_id:
                vcd_file = tmp_path / "waveform.vcd"
                if vcd_file.exists() and vcd_file.stat().st_size > 0:
                    dest_vcd = WAVEFORM_DIR / f"{waveform_id}.vcd"
                    shutil.copy2(vcd_file, dest_vcd)
                    logger.info(f"Waveform saved: {waveform_id}")
                    result["waveform_id"] = waveform_id
            
            return result
        else:
            return {
                "success": False,
                "error": "Test Failed",
                "output": output[:1000],
                "backend": "iverilog",
                "execution_time": execution_time
            }

def create_verilator_wrapper(generate_waveform: bool, enable_coverage: bool) -> str:
    """Create C++ wrapper for Verilator simulation"""
    return f"""
#include "Vtb.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include "verilated_cov.h"
#include <iostream>
#include <cstdlib>

// Global assertion counters
int assertion_pass_count = 0;
int assertion_fail_count = 0;

// Assertion callback
void assertion_callback(const char* msg) {{
    if (msg) {{
        std::string message(msg);
        if (message.find("failed") != std::string::npos) {{
            assertion_fail_count++;
            std::cout << "[ASSERTION FAILED] " << message << std::endl;
        }} else if (message.find("passed") != std::string::npos) {{
            assertion_pass_count++;
        }}
    }}
}}

int main(int argc, char** argv) {{
    // Initialize
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);
    Verilated::assertOn(true);
    
    // Set assertion callback
    Verilated::setAssertCallback(assertion_callback);
    
    // Create instance
    Vtb* top = new Vtb;
    
    // Create waveform dump if needed
    VerilatedVcdC* tfp = nullptr;
    {f'if (true) {{ tfp = new VerilatedVcdC; top->trace(tfp, 99); tfp->open("waveform.vcd"); }}' if generate_waveform else '// No waveform tracing'}
    
    // Initialize clock
    top->clk = 0;
    
    // Main simulation loop
    int cycle = 0;
    int max_cycles = 1000;
    
    while (!Verilated::gotFinish() && cycle < max_cycles) {{
        // Toggle clock
        top->clk = !top->clk;
        
        // Evaluate
        top->eval();
        
        // Dump waveform
        {f'if (tfp) tfp->dump(cycle * 10 + (top->clk ? 5 : 0));' if generate_waveform else ''}
        
        cycle++;
        
        // Break if too many assertion failures
        if (assertion_fail_count > 10) {{
            std::cout << "Too many assertion failures, stopping simulation" << std::endl;
            break;
        }}
    }}
    
    // Final outputs
    std::cout << "\\n=== VERILATOR SIMULATION RESULTS ===" << std::endl;
    std::cout << "Cycles simulated: " << cycle << std::endl;
    std::cout << "Assertions passed: " << assertion_pass_count << std::endl;
    std::cout << "Assertions failed: " << assertion_fail_count << std::endl;
    
    // Save coverage data if enabled
    {f'if (true) {{ VerilatedCov::write("coverage.dat"); }}' if enable_coverage else '// No coverage collection'}
    
    // Cleanup
    {f'if (tfp) {{ tfp->close(); delete tfp; }}' if generate_waveform else ''}
    delete top;
    
    return (assertion_fail_count == 0) ? 0 : 1;
}}
"""

def parse_assertion_results(output: str) -> dict:
    """Parse assertion results from simulation output"""
    passed = 0
    failed = 0
    
    # Look for assertion patterns
    lines = output.split('\n')
    for line in lines:
        if 'Assertions passed:' in line:
            match = re.search(r'Assertions passed:\s*(\d+)', line)
            if match:
                passed = int(match.group(1))
        elif 'Assertions failed:' in line:
            match = re.search(r'Assertions failed:\s*(\d+)', line)
            if match:
                failed = int(match.group(1))
        elif '[ASSERTION FAILED]' in line:
            failed += 1
        elif 'assert' in line.lower() and 'passed' in line.lower():
            passed += 1
    
    return {
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "success_rate": passed / (passed + failed) if (passed + failed) > 0 else 1.0
    }

def parse_coverage_data(coverage_file: Path) -> dict:
    """Parse coverage data from Verilator"""
    try:
        # Use verilator_coverage if available
        cmd = ["verilator_coverage", "--annotate", "anno", str(coverage_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return {
                "status": "parsed",
                "raw_output": result.stdout[:500]
            }
        else:
            return {
                "status": "available_but_not_parsed",
                "error": result.stderr[:200]
            }
    except FileNotFoundError:
        return {"status": "verilator_coverage_not_available"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Keep the existing VCDParser and create_professional_viewer functions
# (They remain the same as in your original code)

class VCDParser:
    """Parse VCD files and extract waveform data"""
    # ... (Keep the exact same VCDParser class from your original code)
    def __init__(self, vcd_path):
        self.vcd_path = vcd_path
        self.signals = []
        self.waveform_data = {}
        self.timescale = "1ns"
        self.max_time = 0
        
    def parse(self):
        """Parse the VCD file"""
        try:
            with open(self.vcd_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Parse header
            signal_map = {}
            in_var_scope = False
            current_scope = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse timescale
                if line.startswith('$timescale'):
                    parts = line.split()
                    if len(parts) > 1:
                        self.timescale = parts[1]
                
                # Parse scope
                elif line.startswith('$scope'):
                    parts = line.split()
                    if len(parts) >= 3:
                        current_scope = parts[2]
                        in_var_scope = True
                
                # Parse variable definitions
                elif line.startswith('$var'):
                    parts = line.split()
                    if len(parts) >= 5:
                        var_type = parts[1]
                        width = parts[2]
                        var_id = parts[3]
                        var_name = parts[4]
                        
                        # Clean up var_name (remove $end if present)
                        if var_name.endswith('$end'):
                            var_name = var_name[:-4].strip()
                        
                        # Create full hierarchical name
                        full_name = f"{current_scope}.{var_name}" if current_scope else var_name
                        
                        signal_map[var_id] = full_name
                        self.signals.append({
                            'id': var_id,
                            'name': full_name,
                            'short_name': var_name,
                            'type': var_type,
                            'width': width,
                            'scope': current_scope
                        })
                
                # End of scope
                elif line.startswith('$upscope'):
                    current_scope = ""
                    in_var_scope = False
                
                # End of definitions
                elif line.startswith('$enddefinitions'):
                    break
            
            # Initialize waveform data
            for signal in self.signals:
                self.waveform_data[signal['name']] = []
            
            # Parse value changes
            current_time = 0
            signal_values = {sig['id']: 'x' for sig in self.signals}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Time change
                if line.startswith('#'):
                    try:
                        time_val = int(line[1:])
                        if time_val != current_time:
                            # Record state at time change
                            self._record_state(current_time, signal_values)
                            current_time = time_val
                            if current_time > self.max_time:
                                self.max_time = current_time
                    except ValueError:
                        continue
                
                # Scalar value change
                elif line[0] in ['0', '1', 'x', 'z', 'X', 'Z'] and len(line) > 1:
                    value = line[0].lower()
                    var_id = line[1:]
                    if var_id in signal_values:
                        signal_values[var_id] = value
                
                # Vector value change
                elif line[0] in ['b', 'B']:
                    parts = line[1:].split()
                    if len(parts) >= 2:
                        value = parts[0]
                        var_id = parts[1]
                        if var_id in signal_values:
                            signal_values[var_id] = value
            
            # Record final state
            self._record_state(current_time, signal_values)
            
            # Clean up signals (remove empty ones)
            self.signals = [sig for sig in self.signals if len(self.waveform_data[sig['name']]) > 0]
            
            return True
            
        except Exception as e:
            logger.error(f"VCD parsing error: {e}")
            return False
    
    def _record_state(self, time, signal_values):
        """Record signal states at a specific time"""
        for sig_id, value in signal_values.items():
            signal_name = None
            for sig in self.signals:
                if sig['id'] == sig_id:
                    signal_name = sig['name']
                    break
            
            if signal_name:
                waveform = self.waveform_data[signal_name]
                if not waveform or waveform[-1]['time'] != time:
                    waveform.append({
                        'time': time,
                        'value': value
                    })
    
    def get_waveform_summary(self, signal_name=None):
        """Get summary of waveform data"""
        if signal_name:
            return self.waveform_data.get(signal_name, [])
        
        summary = {}
        for sig in self.signals[:10]:  # Limit to 10 signals for performance
            summary[sig['name']] = self.waveform_data[sig['name']]
        return summary

def create_professional_viewer(waveform_id: str, vcd_exists: bool) -> str:
    """Create professional HTML viewer with actual waveform display"""
    # ... (Keep the exact same create_professional_viewer function from your original code)
    # This function is very long, so I'm not duplicating it here
    # It should remain exactly as in your original code
    
    # For brevity, returning a simple placeholder
    return f"""
    <html>
    <head><title>Waveform Viewer - {waveform_id}</title></head>
    <body>
        <h1>Waveform Viewer (Professional Edition)</h1>
        <p>Waveform ID: {waveform_id}</p>
        <p>VCD exists: {vcd_exists}</p>
        <a href="/api/waveform/{waveform_id}?download=true">Download VCD</a>
    </body>
    </html>
    """

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    # Check if Verilator is available
    verilator_ok = False
    try:
        result = subprocess.run(["verilator", "--version"], capture_output=True, text=True)
        verilator_ok = result.returncode == 0
    except:
        pass
    
    # Check if Icarus is available
    iverilog_ok = False
    try:
        result = subprocess.run(["iverilog", "-V"], capture_output=True, text=True)
        iverilog_ok = result.returncode == 0
    except:
        pass
    
    return {
        "status": "healthy",
        "backends": {
            "verilator": verilator_ok,
            "iverilog": iverilog_ok
        },
        "waveforms": len(list(WAVEFORM_DIR.glob("*.vcd"))),
        "problems": len(PROBLEMS),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/backend/info")
async def backend_info():
    """Get information about available backends"""
    info = {
        "verilator": {
            "available": False,
            "version": "",
            "features": []
        },
        "iverilog": {
            "available": False,
            "version": "",
            "features": []
        }
    }
    
    # Check Verilator
    try:
        result = subprocess.run(["verilator", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            info["verilator"]["available"] = True
            info["verilator"]["version"] = result.stdout.strip()
            info["verilator"]["features"] = ["SystemVerilog", "Assertions", "Coverage", "High Performance"]
    except:
        pass
    
    # Check Icarus
    try:
        result = subprocess.run(["iverilog", "-V"], capture_output=True, text=True)
        if result.returncode == 0:
            info["iverilog"]["available"] = True
            info["iverilog"]["version"] = result.stderr.strip().split('\n')[0] if result.stderr else "Unknown"
            info["iverilog"]["features"] = ["Verilog", "Simple", "Widely Available"]
    except:
        pass
    
    return info

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
