"""
Global pytest configuration.

Forces all tests to use SQLite regardless of what DATABASE_URL is set to
in .env, ensuring no test ever needs a running PostgreSQL server.
Must run before any app.config import — pytest_configure fires at startup.
"""
import os


def pytest_configure(config):
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_netguard.db"
