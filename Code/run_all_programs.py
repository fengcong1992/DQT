import subprocess
import time

def run_program(program_name):
    print(f"Running {program_name}")
    subprocess.run(["python", program_name])

if __name__ == "__main__":
    programs = ["pull_data","competition_data_process.py", "main.py", "submit.py"]

    for program in programs:
        run_program(program)
        time.sleep(60)  # Add a delay between program executions (if needed)
