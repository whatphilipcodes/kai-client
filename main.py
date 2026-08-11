from src.kai_client.config_client import settings_client


def main():
    print(f"Hello from {settings_client.shared.network.node_id}")


if __name__ == "__main__":
    main()
