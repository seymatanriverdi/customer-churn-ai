"""
MLflow Model Registry yardimcisi: versiyonlari listele, alias tasi.

Kullanim (MLFLOW_TRACKING_URI ayarli olmali):
    python registry_tools.py list
    python registry_tools.py set-alias challenger 2
    python registry_tools.py set-alias champion 2
"""
import sys

from mlflow import MlflowClient

MODEL_NAME = "CustomerChurnModel"

client = MlflowClient()


def list_versions():
    # Alias'lar versiyon aramasinda gelmez; registered model uzerinden okunur.
    registered = client.get_registered_model(MODEL_NAME)
    alias_map = {}
    for alias, alias_version in registered.aliases.items():
        alias_map.setdefault(str(alias_version), []).append(alias)

    versions = sorted(
        client.search_model_versions(f"name='{MODEL_NAME}'"),
        key=lambda v: int(v.version),
    )

    print(
        f"{'ver':<5}{'status':<8}{'aliases':<24}"
        f"{'algorithm':<22}{'recall':>8}{'f1':>8}"
    )
    for v in versions:
        run = client.get_run(v.run_id)
        algorithm = run.data.params.get("algorithm", "-")
        recall = run.data.metrics.get("test_recall", float("nan"))
        f1 = run.data.metrics.get("test_f1", float("nan"))
        aliases = ",".join(sorted(alias_map.get(str(v.version), []))) or "-"
        print(
            f"{v.version:<5}{v.status:<8}{aliases:<24}"
            f"{algorithm:<22}{recall:>8.4f}{f1:>8.4f}"
        )


def set_alias(alias, version):
    client.set_registered_model_alias(MODEL_NAME, alias, version)
    print(f"'{alias}' alias'i Version {version}'a baglandi.")


def main():
    args = sys.argv[1:]

    if args == ["list"]:
        list_versions()
    elif len(args) == 3 and args[0] == "set-alias":
        set_alias(args[1], args[2])
        list_versions()
    else:
        print(__doc__)
        raise SystemExit(1)


if __name__ == "__main__":
    main()