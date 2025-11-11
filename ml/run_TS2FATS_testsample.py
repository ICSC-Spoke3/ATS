import subprocess

def run_command(command):
    print(f"\n Eseguo: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Errore nell'esecuzione: {command}")
        exit(1)
    print(f"Completato: {command}")

def main():
    # Lista delle routine da eseguire in sequenza
    commands = [
        "python FindFlatID.py config/config_flat_test.ini log_flat_test.txt",
        "python findShortSeries.py config/config_shortTS_test.ini log_short_test.txt",
        "python FatsFeatures.py config/config_fats_test.ini log_fats_test.txt"
    ]

    for cmd in commands:
        run_command(cmd)

    print("\n Tutte le routine completate con successo!")

if __name__ == "__main__":
    main()
