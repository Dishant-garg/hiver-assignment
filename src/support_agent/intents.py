"""Intent taxonomy for SpotifyCares, derived from the corpus (see docs/06-intent-taxonomy.md).
Keywords power the weak labeller used for stratified sampling and the simple baseline;
the LLM classifier uses the definitions and examples, not the keywords."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

OTHER = "other"


@dataclass(frozen=True)
class Intent:
    name: str
    definition: str
    examples: tuple[str, ...]
    auto_handle_allowed: bool
    keywords: tuple[str, ...] = field(default_factory=tuple)


# Order matters: weak_label() breaks ties in favour of the earlier intent, so the three
# intents that must escalate are listed first and win any tie against a self-serve intent.
# Definitions carry an explicit "-> sibling" discriminator for the three close pairs
# (billing/subscription, playback/app_bug, content/library), because those are the boundaries
# a labeller — human or LLM — actually gets wrong.
INTENTS: list[Intent] = [
    Intent(
        "account_access",
        "Login/password reset fails, hacked, deletion.",
        ("can't log in, hacked",
         "my account was hacked and I can no longer log in"),
        False,
        ("hacked", "hack", "password", "log in", "login", "logged out", "log into", "sign in",
         "signin", "logging in", "locked out", "username", "recovery email", "reset email",
         "delete my account", "deactivated", "access my account", "account back", "stolen"),
    ),
    Intent(
        "billing_and_refund",
        "Wrong charge, failed payment, refund; plan change -> subscription_plan.",
        ("I cancelled three months ago and you keep taking my money",
         "you charged me twice for over 19 months and I need a refund"),
        False,
        # "charged" is kept alongside "charge": \b-anchored, "charge" does not match "charged".
        # Redundant nesting like "double charged" is dropped -- _distinct_hits() would collapse
        # it into the "charged" span anyway.
        ("charged", "charge", "charges", "charging", "overcharged", "refund", "refunded",
         "billed", "billing", "bill", "invoice", "receipt", "money back", "debited",
         "credit card", "payment", "paypal", "gift card", "giftcard", "my money",
         "prepaid", "price", "paying"),
    ),
    Intent(
        "subscription_plan",
        "Premium/Family/Student/trial plan or cancel; charge -> billing_and_refund.",
        ("can't add son to family plan",
         "it says it can't verify me as a student when I switch plans"),
        False,
        ("student", "family plan", "premium family", "family account", "upgrade", "upgraded",
         "downgrade", "downgraded", "trial", "subscription", "subscribe",
         "unsubscribe", "cancel", "cancelled", "hulu", "unidays", "nus", "renew", "renewal",
         "invite", "invitation", "verify", "verified", "sign up", "signup", "duo", "membership"),
    ),
    Intent(
        "playback_issue",
        "Saved content won't play or skips; crashes -> app_bug.",
        ("not available in offline mode",
         "songs keep skipping after ten seconds and then it stops"),
        True,
        ("won't play", "wont play", "can't play", "cant play", "not playing", "stopped playing",
         "skipping", "skips", "skip", "stutter", "stuttering", "buffering",
         "shuffle", "offline mode", "no sound", "playback", "pausing", "paused", "pauses",
         "repeat", "volume", "queue", "can't listen", "cant listen", "won't let me listen",
         "play my", "playing", "won't stream", "streaming"),
    ),
    Intent(
        "app_bug",
        "Client crashes, freezes or won't open; audio -> playback_issue.",
        ("app crashes on share",
         "the desktop app has been unusable since the last iOS 11 update"),
        True,
        ("crash", "crashes", "crashing", "crashed", "freezes", "frozen", "freezing", "glitch",
         "bug", "error", "broken", "not working", "doesn't work", "does not work", "won't work",
         "wont work", "isn't working", "reinstall", "uninstall", "ios", "android",
         "desktop app", "web player", "battery", "cpu", "lagging", "app won't", "fix the app",
         "update", "updated", "blank screen", "loading"),
    ),
    Intent(
        "library_playlist_issue",
        "Own saved songs/playlists/downloads gone; catalogue -> content_availability.",
        ("my saved songs vanished",
         "the playlist I made on my computer is not appearing on my phone"),
        True,
        ("playlist", "playlists", "saved songs", "my songs", "library", "downloaded", "download",
         "downloads", "deleted", "deleting", "disappeared", "vanished", "lost", "gone", "recover",
         "restore", "sync", "synced", "liked songs", "my music", "song limit", "download limit",
         "10k", "local file", "local files"),
    ),
    Intent(
        "content_availability",
        "Song/album/artist/country never on Spotify or pulled; own saves -> library_playlist_issue.",
        ("album not on Spotify",
         "you need to launch in Kenya, I am tired of using a VPN"),
        True,
        ("not on spotify", "on spotify", "album", "albums", "artist", "artists", "discography",
         "unavailable", "not available", "isn't available", "put back", "taken off", "removed",
         "remove", "launch in", "in my country", "region", "catalogue", "catalog", "licensing",
         "greyed out", "single", "ep", "track", "tracks", "mixtape", "soundtrack", "release",
         "released", "upload"),
    ),
    Intent(
        "feature_request",
        "Wants a feature that doesn't exist or is retired.",
        ("add an Apple Watch app",
         "we would like it if you supported two factor auth"),
        True,
        ("feature", "features", "suggestion", "suggest", "please add", "please make", "wish",
         "would love", "i'd love", "petition", "bring back", "option to", "ability to",
         "apple watch", "iwatch", "lyrics", "two factor", "please bring", "hope you",
         "would be great", "should be able", "add an option", "request"),
    ),
    Intent(
        "how_to_question",
        "Asks how to use a feature that works fine.",
        ("how do I clear the queue",
         "how do the daily mixes work, is it based on my liked songs?"),
        True,
        ("how do", "how to", "how does", "how can i", "how would i", "how i can",
         "is there a way", "where can i", "where do i", "where is the", "which settings",
         "what is the best way", "any way to", "any idea how", "is it possible", "do i have to",
         "what happens", "what will happen", "explain", "steps to", "what does this mean", "what do i do",
         "walk me through", "instructions", "tutorial", "do i need to", "who do i",
         "how long", "how are", "how much"),
    ),
    Intent(
        OTHER,
        "Praise, rants, pitches, or too vague to act on.",
        ("hey I got an issue", "thanks, excellent help"),
        False,
        (),
    ),
]
INTENT_NAMES = [i.name for i in INTENTS]
_BY_NAME = {i.name: i for i in INTENTS}
_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    i.name: [re.compile(r"\b" + re.escape(k) + r"\b") for k in i.keywords] for i in INTENTS
}


def by_name(name: str) -> Intent:
    return _BY_NAME.get(name, _BY_NAME[OTHER])


def _distinct_hits(patterns: list[re.Pattern[str]], text: str) -> int:
    """Number of non-overlapping matched spans.

    Counting one vote per keyword double-counts nested keywords: "double charged" would fire
    both "double charged" and "charged", and "how do i" would fire both "how do i" and
    "how do", inflating an intent purely for having a longer phrase in its list. Collecting
    spans and taking the largest non-overlapping set counts each piece of evidence once.
    """
    spans = [m.span() for p in patterns for m in p.finditer(text)]
    spans.sort(key=lambda s: (s[1], s[0]))
    hits, last_end = 0, -1
    for start, end in spans:
        if start >= last_end:
            hits, last_end = hits + 1, end
    return hits


def weak_label(text: str) -> str:
    """Keyword vote: the intent with the most distinct keyword spans wins; ties -> earlier
    intent; none -> other."""
    if not isinstance(text, str):
        return OTHER
    t = text.lower().replace("’", "'")  # curly apostrophes are common in the corpus
    best, best_hits = OTHER, 0
    for intent in INTENTS:
        hits = _distinct_hits(_PATTERNS[intent.name], t)
        if hits > best_hits:
            best, best_hits = intent.name, hits
    return best


def definitions_block() -> str:
    """Compact text used inside the agent prompt (kept short for the token budget).

    Only the first (deliberately discriminating) example per intent goes into the prompt; the
    second, longer example is kept on the Intent for docs and few-shot use. Two examples per
    intent roughly doubled the block, which lands in every classify call.
    """
    lines = []
    for i in INTENTS:
        ex = "; ".join(f'"{e}"' for e in i.examples[:1])
        lines.append(f"- {i.name}: {i.definition} e.g. {ex}")
    return "\n".join(lines)
