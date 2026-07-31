# Seeds

After `alembic upgrade head`, bootstrap identity and sample corpus with the CLI:

```bash
aegis seed
aegis create-admin --email admin@example.com --password '<strong-password>'
```

`seed.sql` is a placeholder for optional raw-SQL restores; the supported path is `aegis seed`.
