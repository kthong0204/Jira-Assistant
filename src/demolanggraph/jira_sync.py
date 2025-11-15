from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from requests.auth import HTTPBasicAuth

from .settings import SETTINGS, Settings


class JiraSyncError(RuntimeError):
    """Raised when Jira sync configuration or HTTP calls fail."""


class JiraAPIGateway:
    """Minimal Jira client for fetching individual issues."""

    def __init__(
        self,
        base_url: str,
        *,
        email: Optional[str],
        api_token: Optional[str],
        bearer_token: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required for Jira gateway.")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        if bearer_token:
            self._session.headers.update({"Authorization": f"Bearer {bearer_token}"})
        elif email and api_token:
            self._session.auth = HTTPBasicAuth(email, api_token)
        else:
            raise ValueError(
                "Either bearer_token or (email + api_token) must be provided for Jira authentication."
            )

    @property
    def base_url(self) -> str:
        return self._base_url

    def fetch_issue(self, issue_key: str) -> Dict[str, any]:
        url = f"{self._base_url}/rest/api/2/issue/{issue_key}"
        response = self._session.get(url, timeout=self._timeout)
        if response.status_code >= 400:
            raise JiraSyncError(
                f"Jira API error {response.status_code}: {response.text[:200]}"
            )
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise JiraSyncError("Jira API returned non-JSON payload.") from exc


def sync_single_issue(
    issue_key: str,
    settings: Settings | None = None,
) -> Tuple[Path, Dict[str, any]]:
    """Fetch a single Jira issue and append it to the dataset."""

    cfg = settings or SETTINGS
    base_url = cfg.jira_base_url
    if not base_url:
        raise JiraSyncError("Set JIRA_URL in the environment to enable Jira sync.")

    if not (cfg.jira_bearer_token or (cfg.jira_email and cfg.jira_api_token)):
        raise JiraSyncError(
            "Provide either JIRA_BEARER_TOKEN or both JIRA_EMAIL and JIRA_API_TOKEN."
        )

    gateway = JiraAPIGateway(
        base_url,
        email=cfg.jira_email,
        api_token=cfg.jira_api_token,
        bearer_token=cfg.jira_bearer_token,
    )
    print(issue_key)
    payload = gateway.fetch_issue(issue_key)
    fields = payload.get("fields", {}) or {}
    entry = {
        "key": payload.get("key"),
        "summary": fields.get("summary"),
        "description": fields.get("description"),
        "status": (fields.get("status") or {}).get("name"),
        "priority": (fields.get("priority") or {}).get("name"),
        "assignee": (fields.get("assignee") or {}).get("displayName"),
        "reporter": (fields.get("reporter") or {}).get("displayName"),
        "creator": (fields.get("creator") or {}).get("displayName"),
        "issue_type": (fields.get("issuetype") or {}).get("name"),
        "project_key": (fields.get("project") or {}).get("key"),
        "url": f"{gateway.base_url}/browse/{payload.get('key')}",
        "last_updated": fields.get("updated"),
        "comments": (fields.get("comment") or {}).get("comments") or [],
        "custom_fields": {},
    }

    dataset_path = cfg.jira_dataset_path
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_path.exists():
        try:
            data = json.loads(dataset_path.read_text(encoding="utf-8"))
        except Exception:
            data = {"tickets": {}, "summaries": {}, "code_reviews": {}, "test_cases": {}}
    else:
        data = {"tickets": {}, "summaries": {}, "code_reviews": {}, "test_cases": {}}
    tickets = data.setdefault("tickets", {})
    tickets[issue_key] = entry
    dataset_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return dataset_path, entry
