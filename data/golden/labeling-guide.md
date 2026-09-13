# Labelling guide — SpotifyCares golden set

The rules a single labeller applied to every row of `golden_set.csv` (160) and `dev_set.csv`
(40). Written before labelling, applied unchanged; the ten calls that stretched it are
recorded in `docs/03-golden-set.md`.

Each row gets three labels plus a note:

| field | values |
|---|---|
| `intent` | one of the 10 names in `support_agent.intents.INTENT_NAMES` |
| `should_escalate` | `True` / `False` |
| `escalation_reason` | one of `support_agent.rules.ESCALATION_CATEGORIES` |
| `notes` | free text; mandatory on every non-obvious call |

**Invariant:** `should_escalate == (escalation_reason != "none")`. There is no such thing as
an escalation without a reason, or a reason without an escalation. The integrity test
enforces it.

---

## (a) Intents, with two borderline rulings each

The definitions are copied from `docs/06-intent-taxonomy.md`. The rulings under each are the
cases that actually came up and had to be decided once, consistently.

### `account_access` — login/password reset fails, hacked, deletion. *(never auto-handle)*
1. "I got kicked out of Premium and it won't let me back in" → `account_access`, not
   `subscription_plan`. The customer cannot get *in*; the plan is only how they describe
   themselves. Decide on what is broken, not on which product word appears.
2. "Someone else's account is on my email / I want my account deleted" → `account_access`.
   Deletion and identity-ownership are account operations, not `other`, even when phrased as
   a complaint.

### `billing_and_refund` — wrong charge, failed payment, refund. *(never auto-handle)*
1. "I cancelled and you still charged me" → `billing_and_refund`, not `subscription_plan`.
   Money moved after the plan action; the money is the complaint.
2. "My card was declined / payment won't go through" → `billing_and_refund`. A failed
   payment is still a payment problem, even though nothing was charged.

### `subscription_plan` — Premium/Family/Student/trial plan or cancel. *(never auto-handle)*
1. "How do I cancel Premium?" → `subscription_plan`, not `how_to_question`. Cancelling is an
   account operation with a money consequence; the escalate-always class wins the tie.
2. "Student verification won't accept my university" → `subscription_plan`, not
   `app_bug`, even though the customer says "the site is broken". The blocked thing is the
   plan eligibility check.

### `playback_issue` — saved content won't play or skips. *(auto-handle)*
1. "It keeps skipping to the next song after ten seconds" → `playback_issue`, not `app_bug`.
   The client is running; the audio is wrong.
2. "Shuffle isn't random / it only plays the same 20 songs" → `playback_issue`. Playback
   behaviour the customer considers wrong, not a missing feature.

### `app_bug` — client crashes, freezes, won't open or won't update. *(auto-handle)*
1. "The app crashes when I hit share" → `app_bug`, not `feature_request`, even though share
   is a feature. The feature exists and is broken.
2. "Since the iOS update the app won't open" → `app_bug`. Version/update language plus a
   dead client is the canonical case; the brand's reply asks for OS + app version.

### `library_playlist_issue` — the customer's own saved songs/playlists/downloads. *(auto-handle)*
1. "My downloads disappeared after I changed phones" → `library_playlist_issue`, not
   `playback_issue`. The content is gone, not silent.
2. "A playlist I made on desktop isn't showing on my phone" → `library_playlist_issue`.
   Sync of the customer's own content, not a catalogue gap.

### `content_availability` — never on Spotify, pulled, or not in this country. *(auto-handle)*
1. "Why isn't <album> on Spotify" / "when does it drop" → `content_availability`, including
   release-date questions, which are the same catalogue answer.
2. "Launch in Kenya / this is blocked in my region" → `content_availability`. Country
   launches were merged here deliberately (see the taxonomy doc), not into `feature_request`.

### `feature_request` — wants a feature that does not exist or is retired. *(auto-handle)*
1. "Bring back the old library layout" → `feature_request`, not `app_bug`. The customer
   dislikes a shipped change; nothing is broken.
2. "Please make an Apple Watch app" → `feature_request`, not `content_availability`, despite
   "not available".

### `how_to_question` — asks how to use a feature that works fine. *(auto-handle)*
1. "How do I make a collaborative playlist?" → `how_to_question`, not
   `library_playlist_issue`. Nothing is lost; the customer needs the steps.
2. "How do I change my payment method?" → `billing_and_refund`, **not** `how_to_question`.
   A how-to phrased over an account/money operation belongs to the escalate-always intent —
   see the tie-break rules.

### `other` — praise, rants, pitches, or too vague to act on. *(never auto-handle)*
1. "I need help ASAP" with no topic → `other` + `ambiguous`. Urgency is not a subject.
2. "I'm an artist, my song is on the wrong profile" / royalty and earnings questions →
   `other` + `out_of_scope`. A real issue on a different product surface (Spotify for
   Artists); rejected as an intent in the taxonomy because it is under 3% of traffic.

---

## (b) Escalation rules

Escalate (`should_escalate = True`) when **any** of these fires. Reasons are checked in this
order, and the first one that fires is the recorded reason:

1. **`safety_legal`** — legal threats, lawyers, police, fraud/scam accusations, harassment,
   discrimination, self-harm or physical-safety language. Always wins over everything else.
2. **`abusive`** — slurs or sustained abuse directed at the brand or staff. Frustrated
   swearing *about* the product ("this damn app") is not abuse on its own; abuse aimed at a
   person is.
3. **`pii`** — the message contains a card number, email address, or phone number. The
   customer has already exposed data in public; a human must handle it. This fires on the
   *presence* of the data, whatever the topic.
4. **`human_requested`** — the customer explicitly asks for a real person, an agent, a
   supervisor, or to be phoned/DM'd by a human.
5. **`needs_account_access`** — answering would require looking at, or changing, this
   specific account: login, password, hacked, deletion, plan state, student verification,
   family invites, cancellation. Covers all of `account_access` and `subscription_plan`.
6. **`billing_dispute`** — money has moved or is supposed to (charge, double charge, refund,
   failed payment, gift card). Covers `billing_and_refund`.
7. **`out_of_scope`** — not a support request at all: praise, jokes, memes, job pitches,
   partnership/business-development pitches, petitions, artist/royalty matters, questions
   about other companies. There is nothing to fix.
8. **`ambiguous`** — it *is* a support request but there is not enough in it to answer:
   "help", "it's broken", "?", a bare screenshot mention, a reply fragment with no subject.
   **Splitting 7 from 8 on a content-free rant:** if the rant names a product surface that is
   supposedly broken ("your web player sucks", "why is there no help number"), it is a support
   complaint with no detail → `ambiguous`. If it names nothing to fix ("their page is a PR
   stunt", "sort urself out hun") → `out_of_scope`.
9. **The reference-reply rule** — if the brand's own first reply is a DM deflect for account
   details ("can you DM us your username or email") **and carries no published fix**, the
   brand itself judged this message to need a human, and the row escalates. This applies
   **regardless of how clear the intent looks or how obvious the published answer seems to the
   labeller** — the test is what the reply actually did, not what the labeller thinks it could
   have done. A reply that gives a step, a link or an explanation *as well as* asking for a DM
   does not trigger the rule. The reason is the one that best explains why the brand needed
   the account: `needs_account_access` by default, `billing_dispute` when money is involved.
   See (d) on the one thing this rule cannot do: it decides **escalation**, never the intent.

Do **not** escalate (`escalation_reason = "none"`) when the message is a troubleshooting or
information request with a standard published answer: playback problems, app crashes,
library/download sync, "why isn't this on Spotify", how-to questions, feature requests. These
get a first, correct, in-thread step — reinstall, ask for OS + version, link the "Downloads
unexpectedly removed" article, explain licensing, log the idea — exactly as the brand does
in 71–93% of those replies.

## (c) Tie-break rules

1. **Escalate-always beats auto-handle.** If a message is genuinely two intents and one of
   them is `account_access`, `billing_and_refund` or `subscription_plan`, that one wins. The
   cost of a wrong auto-handle on an account or a charge is much higher than the cost of an
   unnecessary handoff. (This matches the deliberate ordering of `INTENTS`.)
2. **Money beats plan.** Both present → `billing_and_refund`.
3. **The broken thing beats the mentioned thing.** Label what fails, not which product noun
   appears: a crash while browsing playlists is `app_bug`, not `library_playlist_issue`.
4. **Whose content?** Never on Spotify / pulled → `content_availability`. The customer's own
   saves → `library_playlist_issue`. Exists, saved, silent → `playback_issue`.
5. **A question about how to do an account or money operation is that operation's intent,**
   not `how_to_question`.
6. **`other` is a decision, not a default.** Use it only for no-support-request or
   no-information messages. If a real intent is recoverable from the text, label it, even
   when the weak labeller said `other`.
7. **`other` never auto-handles.** Every `other` row escalates as `out_of_scope` (nothing to
   fix) or `ambiguous` (too little to act on), unless a higher-priority reason fires.
8. **Language is not an intent.** Non-English messages are labelled on their content like any
   other row, with `non-English` recorded in `notes`. They are not automatically `other` and
   not automatically escalated.
9. **The surface test for non-product questions.** If the request belongs to a different
   Spotify surface or to no Spotify product at all — creator/artist matters (publishing,
   profiles, royalties, playlist pitches), concert presales and ticketing, editorial
   programming, recruitment, partnerships — it is `other` + `out_of_scope`. **Decide this on
   the surface, not on whether the brand happened to answer it publicly.** The brand sometimes
   posts a helpful link for a creator question; that does not make it customer support.
10. **A question about an unsupported or unavailable payment method is
   `billing_and_refund`,** not `feature_request` and not `how_to_question`: "can I pay with a
   gift card / Google Play wallet / phone credit". Money is the subject even when the answer
   is "that method does not exist".
11. **One intent per row.** Multi-issue messages get the intent of the primary complaint —
   the one the customer leads with, or the one the brand answered — with the second issue
   named in `notes`.

## (d) On the reference reply

`reference_reply` is the brand's real first reply from the corpus. It is evidence about what
a competent human did with this message, and it is the deciding evidence for the DM rule in
(b) — but it is **not** the label.

- The label describes the **customer's message**. A reply that is generic ("help's here! what
  device?"), lazy, or simply wrong does not make the message ambiguous or out of scope.
- The brand sometimes DMs a message that has a perfectly good public answer (agent habit,
  shift policy, a thread we cannot see), and sometimes answers in public something that
  really needed the account. Where the message is clear, the message wins.
- **On escalation the reply is decisive, on intent it is not.** A bare DM-for-the-account
  deflect settles `should_escalate` under (b)9 whatever the message looked like — it is the
  only observable standard for "this needed a human" in this dataset. It never sets the
  intent, and a reply that answers publicly never *prevents* an escalation that (b) requires
  or moves a row onto a surface it does not belong to (tie-break 9).
- `weak_label` is a keyword vote with no understanding of the sentence. It was never copied;
  where the final label differs, `notes` says so.
