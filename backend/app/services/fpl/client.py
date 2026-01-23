"""
FPL HTTP Client with Rate Limiting
Handles HTTP requests to FPL API with DefCon rate limiting and exponential backoff.
"""

import httpx
import asyncio
import time
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    DefCon rate limiter with exponential backoff.
    Enforces 60 requests per minute limit with exponential backoff on errors.

    Attributes:
        max_requests: Maximum requests per window
        window_seconds: Time window in seconds
        request_times: List of request timestamps
        backoff_seconds: Current backoff delay
        consecutive_errors: Number of consecutive errors
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window (default: 60)
            window_seconds: Time window in seconds (default: 60)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_times: list[float] = []
        self.backoff_seconds = 0
        self.consecutive_errors = 0
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Wait until rate limit allows a request.

        Applies exponential backoff if errors occurred, then checks
        if we're at the rate limit and waits if necessary.
        """
        async with self.lock:
            now = time.time()

            # Apply exponential backoff if we have errors
            if self.backoff_seconds > 0:
                await asyncio.sleep(self.backoff_seconds)
                self.backoff_seconds = 0  # Reset after waiting

            # Clean old requests outside the window
            self.request_times = [
                t for t in self.request_times if now - t < self.window_seconds
            ]

            # If we're at the limit, wait until the oldest request expires
            if len(self.request_times) >= self.max_requests:
                oldest_request = min(self.request_times)
                wait_time = self.window_seconds - (now - oldest_request) + 0.1
                if wait_time > 0:
                    logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    # Clean again after waiting
                    now = time.time()
                    self.request_times = [
                        t for t in self.request_times if now - t < self.window_seconds
                    ]

            # Record this request
            self.request_times.append(now)

    def record_success(self) -> None:
        """Record a successful request (reset error counter)."""
        self.consecutive_errors = 0

    def record_error(self) -> None:
        """
        Record an error and increase backoff exponentially.

        Exponential backoff: 1s, 2s, 4s, 8s, 16s, max 60s
        """
        self.consecutive_errors += 1
        self.backoff_seconds = min(60, 2 ** (self.consecutive_errors - 1))
        logger.warning(
            f"Rate limiter error #{self.consecutive_errors}, backoff: {self.backoff_seconds}s"
        )


class FPLHTTPClient:
    """
    Async HTTP client for FPL API with rate limiting.

    Handles all HTTP requests to FPL Official API with automatic
    rate limiting, error handling, and retry logic.
    """

    BASE_URL = "https://fantasy.premierleague.com/api"

    def __init__(self, rate_limit_delay: float = 0.1) -> None:
        """
        Initialize HTTP client.

        Args:
            rate_limit_delay: Additional delay between requests in seconds (default: 0.1s)
        """
        self.rate_limit_delay = rate_limit_delay
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        self.rate_limiter = RateLimiter(max_requests=60, window_seconds=60)

    async def get(self, endpoint: str, use_rate_limit: bool = True) -> Dict:
        """
        Make GET request to FPL API with rate limiting.

        Args:
            endpoint: API endpoint (e.g., "bootstrap-static/")
            use_rate_limit: Whether to apply rate limiting (default: True)

        Returns:
            JSON response as dictionary

        Raises:
            httpx.HTTPError: If request fails
        """
        url = f"{self.BASE_URL}/{endpoint}"

        if use_rate_limit:
            await self.rate_limiter.acquire()

        try:
            # Additional delay for rate limiting (legacy support)
            if self.rate_limit_delay > 0:
                await asyncio.sleep(self.rate_limit_delay)

            response = await self.client.get(url)
            response.raise_for_status()

            if use_rate_limit:
                self.rate_limiter.record_success()

            return response.json()

        except httpx.HTTPError as e:
            if use_rate_limit:
                self.rate_limiter.record_error()
            logger.error(f"HTTP request failed for {endpoint}: {str(e)}")
            raise

    async def get_bootstrap_data(self) -> Dict:
        """
        Fetch bootstrap-static data containing all players, teams, and fixtures.

        Returns:
            Dictionary with keys:
            - elements: List of all players
            - teams: List of all teams
            - events: List of gameweeks
            - element_types: Position types
        """
        return await self.get("bootstrap-static/")

    async def get_player_data(self, player_id: int) -> Dict:
        """
        Fetch detailed data for a specific player from element-summary endpoint.

        Args:
            player_id: FPL player ID

        Returns:
            Dictionary containing history, history_past, fixtures, explain
        """
        return await self.get(f"element-summary/{player_id}/")

    async def get_fixtures(self, gameweek: Optional[int] = None) -> Dict:
        """
        Fetch fixtures data.

        Args:
            gameweek: Optional gameweek number to filter fixtures

        Returns:
            Dictionary with fixtures data
        """
        if gameweek:
            return await self.get(f"fixtures/?event={gameweek}")
        return await self.get("fixtures/")

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
