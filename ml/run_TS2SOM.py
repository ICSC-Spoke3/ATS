import subprocess
import os
import argparse

def run_step(description, command):
    """Esegue un comando shell e si ferma in caso di errore."""
    print(f"\n{description}")
    print(f"   → {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Errore durante: {description}")
        exit(1)
    print(f"{description} completato.\n")

def main():
    parser = argparse.ArgumentParser(description="Pipeline completa TS2FATS → SOM → Test (opzionale)")
    parser.add_argument("--skip-test", action="store_true", help="Salta la fase di test finale")
    parser.add_argument("--config-dir", default="config", help="Cartella dove si trovano i file di configurazione")
    args = parser.parse_args()

    config_dir = args.config_dir

    # === Step 1: TS2FATS (già interno) ===
    run_step(
        "Estrazione feature con TS2FATS (FindFlatID + findShortSeries + FatsFeatures)",
        f"python run_TS2FATS.py"
    )

    # === Step 2: Training e salvataggio SOM ===
    run_step(
        "Training e salvataggio SOM",
        f"python training_and_save_som.py {os.path.join(config_dir, 'config_training_som.ini')} log_training_and_save_som.txt"
    )

    # === Step 3: Test (opzionale) ===
    if not args.skip_test:
        test_config = os.path.join(config_dir, "config_feat_to_flag.ini")
        if os.path.exists(test_config):
            run_step(
                "Test da feature a flag (opzionale)",
                f"python test_from_feat_to_flag.py {test_config} log_feat_to_flag.txt"
            )
        else:
            print("File di configurazione per il test non trovato. Fase di test saltata.")
    else:
        print("Fase di test saltata per scelta dell'utente.")

    print("\n Pipeline completata con successo!")

if __name__ == "__main__":
    main()
