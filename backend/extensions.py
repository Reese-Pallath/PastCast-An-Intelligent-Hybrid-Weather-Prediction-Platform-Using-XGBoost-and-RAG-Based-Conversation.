"""
Flask extensions — initialized here, bound to app in create_app().
Import from this module in routes to avoid circular imports.
"""

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="memory://",
    default_limits=["200 per day", "60 per minute"],
)
