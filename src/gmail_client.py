"""Step 2 - Connect to Gmail with OAuth 2.0 (read-only).

Flow:
1. If token.json exists, load the saved login.
2. If the token expired, refresh it silently.
3. Otherwise open the browser once so the user can log in and approve.
4. Save the token so next runs don't ask for login again.
"""
import os

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.config import CREDENTIALS_PATH, GMAIL_SCOPES, TOKEN_PATH


class GmailAuthError(Exception):
    """Raised when we cannot authenticate with Gmail."""


def _load_saved_token():
    """Return saved credentials, or None if missing/corrupted."""
    if not os.path.exists(TOKEN_PATH):
        return None
    try:
        return Credentials.from_authorized_user_file(TOKEN_PATH, GMAIL_SCOPES)
    except ValueError:
        # token.json is broken -> delete it and log in again
        print("[auth] token.json is invalid, logging in again...")
        os.remove(TOKEN_PATH)
        return None


def _login_with_browser():
    """Open the Google consent screen in the browser (first run only)."""
    if not os.path.exists(CREDENTIALS_PATH):
        raise GmailAuthError(
            f"'{CREDENTIALS_PATH}' not found. Download it from Google Cloud "
            "Console > APIs & Services > Credentials (OAuth client, Desktop app)."
        )
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, GMAIL_SCOPES)
    return flow.run_local_server(port=0)


def get_credentials():
    creds = _load_saved_token()

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            print("[auth] Token expired, refreshing...")
            creds.refresh(Request())
        except RefreshError:
            print("[auth] Refresh failed, logging in again...")
            creds = _login_with_browser()
    else:
        creds = _login_with_browser()

    with open(TOKEN_PATH, "w") as f:
        f.write(creds.to_json())
    return creds


def get_gmail_service():
    """Return a ready-to-use Gmail API service object."""
    try:
        return build("gmail", "v1", credentials=get_credentials())
    except HttpError as e:
        raise GmailAuthError(f"Gmail API error: {e}") from e


def test_connection():
    """Print the connected email address to prove the connection works."""
    service = get_gmail_service()
    profile = service.users().getProfile(userId="me").execute()
    print(f"[ok] Connected to: {profile['emailAddress']}")
    print(f"[ok] Total messages: {profile['messagesTotal']}")
    return service


if __name__ == "__main__":
    test_connection()
