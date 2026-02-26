"""Lightweight authentication scaffolding for Streamlit apps.

This prototype simulates role-based access by validating credentials against
environment-provided secrets. The pattern can be swapped for enterprise SSO
once an identity provider is integrated.
"""

import os
from typing import Dict, Optional

import streamlit as st


DEFAULT_TOKENS = [
    ("admin", "admin123", "admin"),
    ("analyst", "analyst123", "analyst"),
    ("viewer", "viewer123", "viewer"),
]


def _load_token_map() -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    env_tokens = os.getenv("ESG_AI_ACCESS_TOKENS")
    if env_tokens:
        for entry in env_tokens.split(","):
            parts = [p.strip() for p in entry.split(":") if p.strip()]
            if len(parts) >= 3:
                username, token, role = parts[:3]
                mapping[username.lower()] = {
                    "username": username,
                    "token": token,
                    "role": role,
                }

    if not mapping:
        # Fallback defaults with env overrides
        for username, token, role in DEFAULT_TOKENS:
            env_key = f"ESG_AI_{role.upper()}_TOKEN"
            mapping[username] = {
                "username": username,
                "token": os.getenv(env_key, token),
                "role": role,
            }

    super_token = os.getenv("ESG_AI_SUPER_TOKEN")
    if super_token:
        mapping["superuser"] = {
            "username": "superuser",
            "token": super_token,
            "role": "admin",
        }

    return mapping


def _verify_credentials(username: str, token: str, mapping: Dict[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
    if not username or not token:
        return None
    user_record = mapping.get(username.lower())
    if not user_record:
        return None
    if token != user_record.get("token"):
        return None
    return {"username": user_record["username"], "role": user_record["role"]}


def _safe_rerun() -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return
    rerun = getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()


def logout() -> None:
    st.session_state.pop("auth_user", None)


def render_user_controls(user: Dict[str, str]) -> None:
    with st.sidebar:
        # Professional user display
        role_colors = {
            "admin": "linear-gradient(135deg, #1e1e1e 0%, #333333 100%)",
            "analyst": "linear-gradient(135deg, #1e1e1e 0%, #333333 100%)",
            "viewer": "linear-gradient(135deg, #1e1e1e 0%, #333333 100%)"
        }
        
        role = user.get('role', 'viewer')
        color = role_colors.get(role, role_colors['viewer'])
        
        st.markdown(f"""
        <div style="
            padding: 1.25rem;
            border-radius: 12px;
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 2rem;
            backdrop-filter: blur(8px);
        ">
            <div style="font-family: 'Inter', sans-serif;">
                <div style="font-weight: 700; color: #ffffff; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;">
                    Active Session
                </div>
                <div style="font-weight: 700; color: #ffffff; font-size: 1.1rem; margin: 0.25rem 0;">
                    {user.get('username', 'Unknown')}
                </div>
                <div style="
                    display: inline-block;
                    padding: 0.2rem 0.6rem;
                    border-radius: 6px;
                    font-size: 0.65rem;
                    font-weight: 800;
                    background: {color};
                    color: #ffffff;
                    border: 1px solid rgba(255, 255, 255, 0.3);
                    margin-top: 0.5rem;
                    text-transform: uppercase;
                ">{role} access</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("SIGN OUT", key="auth_logout", use_container_width=True):
            logout()
            _safe_rerun()


def ensure_authenticated(allowed_roles: Optional[list] = None) -> Optional[Dict[str, str]]:
    """Ensure user is authenticated. Returns user dict or None if not authenticated."""
    
    # Inject login styling
    st.markdown("""
    <style>
    .auth-help-text {
        font-size: 0.75rem;
        color: #ffffff;
        margin-bottom: 0.75rem;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)
    
    mapping = _load_token_map()
    user = st.session_state.get("auth_user")
    
    # Check if already authenticated
    if user:
        if allowed_roles and user.get("role") not in allowed_roles:
            st.sidebar.error("Access Denied: Insufficient Privileges")
            logout()
            _safe_rerun()
            return None
        return user

    # Show login form with professional UI
    with st.sidebar:
        st.markdown("---")
        
        # Login header
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h2 style="margin: 0; color: #fff; font-size: 1.25rem; font-weight: 700; letter-spacing: -0.01em;">Gateway Access</h2>
            <p style="margin: 0.5rem 0; color: #cbd5e1; font-size: 0.85rem; font-weight: 500;">Secure Authentication Required</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Username field
        username = st.text_input("USER", key="auth_username", placeholder="Username")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Password field
        token = st.text_input("CORE ACCESS CODE", type="password", key="auth_token", placeholder="Access code")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Buttons
        submitted = st.button("SIGN IN", key="auth_submit", use_container_width=True, type="primary")
        
        demo = st.button("DEMO ACCESS", key="auth_demo", use_container_width=True)
        
        if demo:
            st.session_state.auth_username = "admin"
            st.session_state.auth_token = "admin123"
            st.info("Demo credentials loaded. Press AUTHENTICATE.")
            _safe_rerun()
        
        if submitted:
            if username and token:
                with st.spinner("VALIDATING..."):
                    verified = _verify_credentials(username, token, mapping)
                    if verified:
                        st.session_state["auth_user"] = verified
                        st.success(f"Access Granted: {verified['username']}")
                        _safe_rerun()
                    else:
                        st.error("Authentication Failed: Invalid Credentials")
            else:
                st.warning("Input Required: Missing Credentials")
        
        # Help hint
        st.markdown("""
        <div style="
            background: rgba(30, 41, 59, 0.4);
            padding: 1.25rem;
            border-radius: 8px;
            margin-top: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
        ">
            <div style="font-size: 0.75rem; color: #ffffff; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                Access Credentials:<br>
                ROOT: admin / admin123
            </div>
        </div>
        """, unsafe_allow_html=True)

    return None


__all__ = ["ensure_authenticated", "render_user_controls", "logout", "_load_token_map", "_verify_credentials", "_safe_rerun"]
