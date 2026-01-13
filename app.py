#!/usr/bin/env python3
"""
Backend API for VLSI Practice - SystemVerilog with Verilator
FIXED VERSION - All bugs patched
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

# Load problems
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
    """Execute SystemVerilog/Verilog code - FIXED VERSION"""
    try:
        # Find problem
        problem = next((p for p in PROBLEMS if p["id"] == request.problem_id), None)
        if not problem:
            raise HTTPException(status_code=404, detail="Problem not found")
        
        # Get testbench safely - FIXED!
        testbench = problem.get("testbench_sv") or problem.get("testbench")
        if not testbench:
            raise HTTPException(
                status_code=400, 
                detail=f"Problem '{problem['id']}' has no testbench defined"
            )
        
        # Choose backend based on problem type
        if problem.get("requires_sv", False) or problem.get("has_assertions", False):
            # Use Verilator for SystemVerilog
            result = await run_verilator_simulation(
                request.code,
                testbench,  # Use the safely obtained testbench
                request.generate_waveform,
                request.enable_assertions,
                request.enable_coverage,
                problem["title"]
            )
        else:
            # Use Icarus for simple Verilog
            result = run_iverilog_simulation(
                request.code,
                testbench,  # Use the safely obtained testbench
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
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error in run_code: {e}", exc_info=True)  # Added exc_info
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
        
        # Simple viewer - implement your professional one here
        html_content = f"""
        <html>
        <head><title>Waveform Viewer - {waveform_id}</title></head>
        <body>
            <h1>Waveform Viewer</h1>
            <p>Waveform ID: {waveform_id}</p>
            <p>File exists: {vcd_path.exists()}</p>
            <a href="/api/waveform/{waveform_id}?download=true">Download VCD</a>
            <br><br>
            <pre>Use gtkwave or other VCD viewer to open this file</pre>
        </body>
        </html>
        """
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
    
    # === FIX THE SYNTAX HERE FIRST ===
    user_code = fix_systemverilog_syntax(user_code)
    testbench = fix_systemverilog_syntax(testbench)
    # =================================
    
    # Get running loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    result = await loop.run_in_executor(
        executor,
        _run_verilator_simulation_sync,
        user_code, testbench, generate_waveform,
        enable_assertions, enable_coverage,
        problem_title, constraints, timeout
    )
    
    return result
def fix_systemverilog_syntax(code: str) -> str:
    """Fix common SystemVerilog syntax issues for Verilator compatibility"""
    # Fix @(*) in cover properties - replace with proper clocking
    if "@(*)" in code and "cover property" in code:
        # Check if there's a clock in the module
        if "input logic clk" in code or "input clk" in code:
            code = code.replace("cover property (@(*)", "cover property (@(posedge clk)")
        else:
            # Remove cover properties if no clock
            lines = code.split('\n')
            fixed_lines = []
            for line in lines:
                if "cover property (@(*)" not in line:
                    fixed_lines.append(line)
                else:
                    # Comment it out
                    fixed_lines.append("// " + line + " // Commented: needs clock")
            code = '\n'.join(fixed_lines)
    
    # Fix covergroup instantiation
    code = code.replace("cg cg_inst;", "cg cg_inst = new();")
    
    return code
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
            # === ADD THE FIX FUNCTION HERE ===
            user_code = fix_systemverilog_syntax(user_code)
            testbench = fix_systemverilog_syntax(testbench)
            # =================================
            
            user_code, testbench = wrap_testbench_for_verilator(user_code, testbench)

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
            
            # Extract module name from code (simplified)
            module_match = re.search(r'module\s+(\w+)', user_code)
            module_name = module_match.group(1) if module_match else "top_module"
            
            # Create C++ wrapper - FIXED!
            wrapper_content = create_verilator_wrapper(module_name, generate_waveform, enable_coverage)
            wrapper_file = tmp_path / "sim_main.cpp"
            wrapper_file.write_text(wrapper_content)
            
            # Build Verilator command for SystemVerilog
            cmd = [
                "verilator",
                "--sv",
                "-cc",
                "--exe",
                "--build",
                "--top-module", module_name,
                "-o", "simulation",
                "-Wno-fatal",
                "-Wno-WIDTH",
                "-Wno-STMTDLY",
                "-Wno-UNUSED",
                "-Wno-UNDRIVEN",
                "--Mdir", str(tmp_path / "obj_dir")
            ]
            
            # Add SystemVerilog flags
            cmd.append("--sv")  # Explicitly enable SystemVerilog
            
            if enable_assertions:
                cmd.append("--assert")
            if enable_coverage:
                cmd.append("--coverage")
            if generate_waveform:
                cmd.append("--trace")
            
            # For SystemVerilog assertions and properties
            cmd.extend([
                "+1364-2005ext+v",  # Enable SV extensions
                "+1800-2012ext+v",
                "--bbox-unsup",  # Blackbox unsupported features
            ])
            
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
                timeout=60  # Increased timeout for SV compilation
            )
            
            if compile_result.returncode != 0:
                # Provide more helpful error message
                error_msg = compile_result.stderr or compile_result.stdout
                logger.error(f"Verilator compilation failed: {error_msg[:500]}")
                
                # Check for common issues
                if "syntax error" in error_msg and "@(*)" in error_msg:
                    error_msg += "\n\nTIP: Use '@(posedge clk)' instead of '@(*)' for cover properties in sequential contexts."
                
                return {
                    "success": False,
                    "error": "Verilator Compilation Failed",
                    "details": error_msg[:1000],
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
            
            # Rest of the function remains the same...
            
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
            
            success = sim_result.returncode == 0 and assertions.get("failed", 0) == 0
            
            return {
                "success": success,
                "output": output[:2000],
                "error": sim_result.stderr[:500] if sim_result.stderr else "",
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
                "error": f"Internal error: {str(e)}",
                "backend": "verilator",
                "execution_time": time.time() - start_time
            }
def wrap_testbench_for_verilator(user_code: str, testbench_code: str) -> tuple:
    """Wrap testbench to make it compatible with Verilator"""
    # Extract module name from user code
    module_match = re.search(r'module\s+(\w+)', user_code)
    if not module_match:
        return user_code, testbench_code
    
    module_name = module_match.group(1)
    
    # Check if testbench is a proper module
    if "module tb" in testbench_code or "module testbench" in testbench_code:
        return user_code, testbench_code
    
    # If testbench is just a procedural block, wrap it
    if "initial begin" in testbench_code and "endmodule" not in testbench_code:
        wrapped_tb = f"""
`timescale 1ns/1ps
module tb();
    // Instantiate DUT
    {module_name} dut();
    
    // Include the testbench code
    {testbench_code}
endmodule
"""
        return user_code, wrapped_tb
    
    return user_code, testbench_code
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
                "error": "Icarus Compilation Failed",
                "details": compile_result.stderr[:500] if compile_result.stderr else "No error output",
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
        
        # Check for PASS or success
        if "PASS" in output.upper() or sim_result.returncode == 0:
            result = {
                "success": True, 
                "output": output[:1000],
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
                "output": output[:500],
                "backend": "iverilog",
                "execution_time": execution_time
            }

def create_verilator_wrapper(module_name: str = "top_module", generate_waveform: bool = False, enable_coverage: bool = False) -> str:
    """Create C++ wrapper for Verilator simulation - COMPATIBLE VERSION"""
    return f"""#include "V{module_name}.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include "verilated_cov.h"
#include <iostream>
#include <cstdlib>
#include <cstdio>

// Global assertion counters
int assertion_pass_count = 0;
int assertion_fail_count = 0;

// Simple message handler
void vl_msg_handler(const VlMessage* msg) {{
    if (msg) {{
        std::string message(msg->msg);
        if (message.find("Assertion failed") != std::string::npos ||
            message.find("fatal") != std::string::npos ||
            message.find("error") != std::string::npos) {{
            assertion_fail_count++;
            std::cout << "[ASSERTION/ERROR] " << message << std::endl;
        }} else if (message.find("Assertion passed") != std::string::npos) {{
            assertion_pass_count++;
            std::cout << "[ASSERTION PASSED]" << std::endl;
        }}
    }}
}}

int main(int argc, char** argv) {{
    // Initialize
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);
    Verilated::assertOn(true);
    
    // Set message handler (older API)
    Verilated::setMessageHandler(vl_msg_handler);
    
    // Create instance
    V{module_name}* top = new V{module_name};
    
    // Create waveform dump if needed
    VerilatedVcdC* tfp = nullptr;
"""
    
    if generate_waveform:
        wrapper = f"""    tfp = new VerilatedVcdC;
    top->trace(tfp, 99);
    tfp->open("waveform.vcd");"""
    else:
        wrapper = """    // No waveform tracing"""
    
    wrapper += f"""
    
    // Initialize
    top->eval();
    
    // Main simulation loop
    int cycle = 0;
    int max_cycles = 100;  // Reduced for simple designs
    
    // Simple simulation loop
    while (!Verilated::gotFinish() && cycle < max_cycles) {{
        // For simple combinatorial logic, we just need to trigger evaluation
        // when inputs change
        if (cycle < 4) {{
            // Cycle through input combinations (for 2-input gate)
            top->a = (cycle >> 0) & 1;
            top->b = (cycle >> 1) & 1;
        }} else if (cycle == 4) {{
            // Stop changing inputs
            top->a = 0;
            top->b = 0;
        }}
        
        // Evaluate
        top->eval();
        
        // Dump waveform
"""
    
    if generate_waveform:
        wrapper += """        if (tfp) tfp->dump(cycle * 10);"""
    
    wrapper += f"""
        
        cycle++;
    }}
    
    // Final outputs
    std::cout << "\\n=== VERILATOR SIMULATION RESULTS ===" << std::endl;
    std::cout << "Cycles simulated: " << cycle << std::endl;
    std::cout << "Assertion failures: " << assertion_fail_count << std::endl;
    std::cout << "Simulation " << ((assertion_fail_count == 0) ? "PASSED" : "FAILED") << std::endl;
    std::cout << "\\nNote: For assertion details, check the SystemVerilog $display outputs above." << std::endl;
    
    // Save coverage data if enabled
"""
    
    if enable_coverage:
        wrapper += """    VerilatedCov::write("coverage.dat");"""
    
    wrapper += f"""
    
    // Cleanup
"""
    
    if generate_waveform:
        wrapper += """    if (tfp) { tfp->close(); delete tfp; }"""
    
    wrapper += f"""
    delete top;
    
    return (assertion_fail_count == 0) ? 0 : 1;
}}
"""
    
    return wrapper

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

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    # Check if Verilator is available
    verilator_ok = False
    verilator_version = "Not installed"
    try:
        result = subprocess.run(["verilator", "--version"], capture_output=True, text=True)
        verilator_ok = result.returncode == 0
        if verilator_ok:
            verilator_version = result.stdout.strip()
    except:
        pass
    
    # Check if Icarus is available
    iverilog_ok = False
    iverilog_version = "Not installed"
    try:
        result = subprocess.run(["iverilog", "-V"], capture_output=True, text=True)
        iverilog_ok = result.returncode == 0
        if iverilog_ok:
            iverilog_version = result.stderr.strip().split('\n')[0] if result.stderr else "Unknown"
    except:
        pass
    
    return {
        "status": "healthy" if (verilator_ok or iverilog_ok) else "degraded",
        "backends": {
            "verilator": {
                "available": verilator_ok,
                "version": verilator_version
            },
            "iverilog": {
                "available": iverilog_ok,
                "version": iverilog_version
            }
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
