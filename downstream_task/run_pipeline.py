import subprocess

subprocess.run(["python", "stage_b_generate_critique.py"], check=True)
subprocess.run(["python", "stage_c_refine_response.py"], check=True)
