# master run.py
import subprocess

def run_script(script_name):
    print(f"Running {script_name}...")
    result = subprocess.run(['python', script_name], capture_output=True, text=True)

    print(result.stdout)

    if result.stderr:
        print(f"--- STDERR ---\n{result.stderr}\n")

    if result.returncode != 0:
        print(f"Error running {script_name}, exit code {result.returncode}. Continuing to next script.\n")
        # Uncomment the next line to stop execution on error
        # exit(result.returncode)
    else:
        print(f"{script_name} finished successfully.\n")

if __name__ == "__main__":
    scripts = ['raw_data.py', 'staging.py', 'presentation.py']
    for script in scripts:
        run_script(script)

    print("All scripts attempted to run.")