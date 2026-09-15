// English translations for review cards: stored ones come with API responses, missing
// ones are requested from POST /api/translations and cached for the page's lifetime.

import { post } from "./api.js";
import { escapeHtml, languageName, toast } from "./ui.js";

const TARGET_LANGUAGE = "en";
const BATCH_SIZE = 50;
const cache = new Map(); // review id → { outcome, text }
const inFlight = new Set();

/** Cache translations that arrived with reviews from the API. */
export function rememberTranslations(reviews) {
  for (const review of reviews) {
    if (review.translation) cache.set(review.review_id, { outcome: "cached", text: review.translation });
  }
}

/** Render translation state for every card under ``root``. */
export function refreshTranslations(root) {
  renderAll();
}

/** Handle the per-card translation buttons. */
export function handleTranslationAction(button) {
  const card = button.closest(".review[data-review-id]");
  if (!card) return;
  const action = button.dataset.action;
  if (action === "show-original") {
    card.dataset.view = "original";
  } else if (action === "show-translation" || action === "translate") {
    card.dataset.view = "translation";
    void translate([card]);
  }
  renderAll();
}

const reviewCards = (root) => [...root.querySelectorAll(".review[data-review-id]")];
const reviewId = (card) => Number(card.dataset.reviewId);
const needsTranslation = (card) =>
  Boolean(card.dataset.language) && card.dataset.language !== TARGET_LANGUAGE;

async function translate(cards) {
  const ids = [
    ...new Set(cards.map(reviewId).filter((id) => !cache.has(id) && !inFlight.has(id))),
  ];
  if (!ids.length) return;
  ids.forEach((id) => inFlight.add(id));
  renderAll();

  const slowNotice = setTimeout(
    () => toast("Loading the local translation model. The first request takes a few seconds…"),
    1500,
  );
  try {
    for (let start = 0; start < ids.length; start += BATCH_SIZE) {
      const response = await post("/api/translations", { review_ids: ids.slice(start, start + BATCH_SIZE) });
      for (const item of response.items) cache.set(item.review_id, { outcome: item.outcome, text: item.text });
    }
  } catch (error) {
    for (const id of ids) if (!cache.has(id)) cache.set(id, { outcome: "error", text: null });
    toast(`Translation failed: ${error.message}`);
  } finally {
    clearTimeout(slowNotice);
    ids.forEach((id) => inFlight.delete(id));
    renderAll();
  }
}

function renderAll() {
  for (const card of reviewCards(document)) renderCard(card);
}

function renderCard(card) {
  const original = card.querySelector('[data-role="original"]');
  const translated = card.querySelector('[data-role="translation"]');
  const status = card.querySelector('[data-role="translation-status"]');
  if (!needsTranslation(card)) {
    original.hidden = false;
    translated.hidden = true;
    status.hidden = true;
    return;
  }

  const id = reviewId(card);
  const entry = cache.get(id);
  const source = escapeHtml(languageName(card.dataset.language));
  const wantsTranslation = card.dataset.view === "translation";
  status.hidden = false;

  if (entry?.text) {
    translated.textContent = entry.text;
    translated.hidden = !wantsTranslation;
    original.hidden = wantsTranslation;
    status.innerHTML = wantsTranslation
      ? `Translated from ${source} · <button type="button" class="text-button" data-action="show-original">Show original</button>`
      : `${source} · <button type="button" class="text-button" data-action="show-translation">Show English</button>`;
    return;
  }

  original.hidden = false;
  translated.hidden = true;
  if (inFlight.has(id)) {
    status.innerHTML = `<span class="spinner small" aria-hidden="true"></span> Translating from ${source}…`;
  } else if (entry) {
    status.textContent = `${languageName(card.dataset.language)} · translation unavailable`;
  } else {
    status.innerHTML = `${source} · <button type="button" class="text-button" data-action="translate">Translate to English</button>`;
  }
}
