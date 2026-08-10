# Security and credentials

Do not place API keys, Hugging Face tokens, cluster credentials, private model paths, or W&B keys in source, configs, logs, issues, or pull requests.

Use environment variables or the cluster secret store. `.env` files are ignored. Rotate any credential that has been pasted into chat, logs, or a committed file.

Report accidental credential exposure by removing the secret from the repository history where necessary, rotating it immediately, and recording only the remediation—not the secret—in the decision log.
