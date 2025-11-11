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
        "python FindFlatID.py config/config_flat.ini log_flat.txt",
        "python findShortSeries.py config/config_shortTS.ini log_short.txt",
        "python FatsFeatures.py config/config_fats.ini log_fats.txt"
    ]

    for cmd in commands:
        run_command(cmd)

    print("\n Tutte le routine completate con successo!")

if __name__ == "__main__":
    main()
