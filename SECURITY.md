# Security

The default deployment binds the gateway, model services, and TURN listener
to loopback. Use an SSH tunnel for access from a laptop. For HTTPS or public
access, follow [the deployment guide](docs/deployment.md).

Set a long random `AUTH_TOKEN` before sharing the gateway through a proxy,
tunnel, or LAN listener. The gateway requires a token for non-loopback binds.
The token is shared by all users and stored in browser local storage. It does
not provide individual accounts or session ownership. Put an identity-aware
proxy in front when individual access control is needed.

`ALLOWED_ORIGINS` controls cross-origin browser access. State-changing
requests with an unlisted browser origin are rejected. Origin checks do not
authenticate scripts or other HTTP clients. Keep inference and synthesis
ports private because those services do not use the gateway's token.

Recognition, generation, and synthesis run on the configured host services.
Installation and model downloads use external services. The default STUN
service receives network metadata. A configured TURN service carries audio.
Runtime logs can contain recognized text and network addresses. Review logs
before attaching them to an issue. Conversation history lives in gateway
memory and is cleared when a session closes.

Local `.env` files and runtime directories are excluded from Git and Docker
build contexts. Compose creates TURN credential files with owner-only
permissions. Keep backups and copied config files private as well. If a
credential is exposed, replace it. Deleting a file does not invalidate the
credential or remove it from Git history.

Report a suspected vulnerability privately to a repository maintainer. Use
GitHub's **Report a vulnerability** option when it is enabled. Otherwise
request a private reporting channel without posting the exploit or credentials
in a public issue. Include the affected version, reproduction steps, and impact.
