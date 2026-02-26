"""Authentication and user context dependencies."""

from typing import Dict, Optional

from fastapi import Depends, Header, HTTPException, status

from utils.logging import get_logger

LOGGER = get_logger(__name__)


def get_current_user(
    x_user: Optional[str] = Header(None, alias="X-User"),
    x_role: Optional[str] = Header(None, alias="X-Role"),
) -> Dict[str, str]:
    if not x_user or not x_role:
        LOGGER.warning("missing_auth_headers", extra={"user": x_user, "role": x_role})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return {"username": x_user, "role": x_role}


CurrentUser = Depends(get_current_user)


__all__ = ["get_current_user", "CurrentUser"]
