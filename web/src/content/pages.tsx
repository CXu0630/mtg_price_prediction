import type { ReactNode } from 'react'
import explore from './explore.json'
import { Bars, Histogram, PairedBars, Stats, Table, count, money } from './charts'

/* The documentation-style pages. Their text is drawn from this repo's README, docs/APP_DESIGN.md and model
   cards, and from the teammate repo's README and reports (pwoodsCMU/mtgpricepredictor: reports/A3_buylist_bot.md,
   reports/C_final_test.md, reports/E_bot_rules.md). The exploration numbers come from
   scripts/app/build_explore_stats.py (-> explore.json). Update them together. */

export type PageId = 'predictor' | 'docs' | 'bots' | 'data'

export const PAGES: { id: PageId; label: string; title: string; eyebrow: string; lede: string }[] = [
  { id: 'predictor', label: 'Predictor', title: 'MTG Price Intelligence', eyebrow: 'Magic: The Gathering',
    lede: '' },
  { id: 'docs', label: 'Docs', title: 'Documentation', eyebrow: 'How it works',
    lede: 'What each number on a card means, which model produces it, how far to trust it, and how the site is built and kept fresh.' },
  { id: 'bots', label: 'Bot Trading', title: 'Bot Trading', eyebrow: 'Paper trading',
    lede: 'Do the forecasts make money at prices you can actually trade at? A paper-trading bot with $100 of fake money, real costs and no look-ahead says: not reliably.' },
  { id: 'data', label: 'Data Exploration', title: 'Data Exploration', eyebrow: 'The dataset',
    lede: `What ${count(explore.totals.priced)} priced printings look like: where the money is, what makes a card expensive, and how prices moved over one summer.` },
]
export const pageMeta = (id: PageId) => PAGES.find((p) => p.id === id)!

export interface Block { key: string; title?: string; body: ReactNode; note?: ReactNode }

const pct = (v: number) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(1)}%`

// ---------------------------------------------------------------- Documentation
const DOCS: Block[] = [
  { key: 'what', title: 'What you get for each card', body: (
    <dl className="doc-defs">
      <dt>Fair value</dt>
      <dd>What the printing should cost given what it is (rarity, finish, treatments, rules text, legality,
        demand, reprints, supply), with an 80% range. From the boosted-tree model by default; the Boosted trees /
        Deep GP switch on the Predictor page swaps in the deep Gaussian process.</dd>
      <dt>Current</dt><dd>TCGplayer's market price on the latest snapshot day.</dd>
      <dt>In 14 days</dt>
      <dd>The TCGplayer forecast: where the TCGplayer price will be two weeks out, with a calibrated 80% range and a signal.</dd>
      <dt>CK buylist</dt><dd>The buylist forecast: what Card Kingdom's buylist will pay in 14 days.</dd>
      <dt>Similar cards</dt><dd>The five printings the deep GP places closest to this one: “priced like these”.</dd>
    </dl>
  ) },
  { key: 'signals', title: 'BUY, SELL and HOLD', body: (
    <dl className="doc-defs">
      <dt>BUY</dt><dd>The whole 80% range for the 14-day price sits above today's price.</dd>
      <dt>SELL</dt><dd>The whole range sits below today's price.</dd>
      <dt>HOLD</dt><dd>Anything else, and always when the price more than doubled in the last week without
        confirmation from other markets (an unconfirmed jump).</dd>
    </dl>
  ), note: 'A signal is about TCGplayer\'s market price, a lagging sales average, not a price you can buy at. See Bot Trading.' },
  { key: 'models', title: 'The models', body: (
    <Table head={['', 'Question', 'Learns from', 'Gives']} rows={[
      ['Boosted trees (fair value)', 'What should this printing cost, from what it is?',
        'Attributes and supply: rarity, finish, treatments, rules-text tags, legality, EDHREC rank, age, reprints, set size, pull rates. No price history.',
        'Fair value and 80% range'],
      ['Deep GP (fair value)', 'Same question, a second opinion',
        '~465 tabular features plus oracle-text (all-mpnet-base-v2) and card-art (CLIP) embeddings, through a sparse variational Gaussian process',
        'Fair value, 80% range and similar cards'],
      ['TCGplayer forecast', 'Where will the TCGplayer price be in 14 days?',
        'Momentum, volatility, release timing, EDHREC rank, moves of the card\'s other printings and its set, CK / Cardmarket levels vs TCGplayer',
        'Median and calibrated 80% range; BUY / SELL / HOLD'],
      ['Buylist forecast', 'What will Card Kingdom\'s buylist pay in 14 days?',
        'The TCGplayer forecast\'s inputs plus the buylist\'s own 7- and 21-day change and its level vs CK retail',
        'Median and calibrated 80% range'],
    ]} />
  ), note: 'The fair-value model and both forecasts are gradient-boosted trees (the forecasts are quantile models with conformal calibration), retrained here from the teammate\'s published method, where they are called Models A, B and C. The forecasts were retrained without the fair-value model\'s inputs, so switching fair-value models never changes a forecast.' },
  { key: 'accuracy-a', title: 'How accurate is fair value?', body: (
    <Table head={['80% range covers the true price', 'CatBoost', 'Deep GP']} align={['l', 'r', 'r']} rows={[
      ['under $1', '77.6%', '83.9%'], ['$1–10', '82.9%', '75.4%'], ['$10–100', '71.8%', '76.0%'],
      ['$100+', '52.3%', '67.6%'], ['overall', '78.1%', '80.5%'], ['typical error (log MAE)', '0.52', '0.57'],
    ]} />
  ), note: 'Held-out cards never seen in training, split by card. The deep GP\'s ranges hold up better on expensive cards; CatBoost (a close cousin of the boosted-tree model) has the sharper midpoint. The boosted-tree model itself: out-of-fold log error 0.44, 80% coverage 80.0%.' },
  { key: 'accuracy-b', title: 'How accurate are the 14-day forecasts?', body: (
    <Table head={['Final test, Sep 14–18', 'Rank corr.', 'BUY right', 'SELL right', '80% coverage']}
      align={['l', 'r', 'r', 'r', 'r']} rows={[
        ['TCGplayer forecast, all printings', '0.215', '70.4%', '89.8%', '81.1%'],
        ['TCGplayer forecast, held-out cards', '0.228', '75.9%', '85.2%', '81.3%'],
        ['Buylist forecast, all printings', '0.225', '69.6%', '58.6%', '78.1%'],
        ['Buylist forecast, held-out cards', '0.243', '62.0%', '38.1%', '77.5%'],
      ]} />
  ), note: 'Scored once, never tuned on: models trained on data up to Aug 23, outcomes through Oct 2. They rank moves usefully but modestly; a BUY gains about 3.5% on average.' },
  { key: 'custom', title: 'Custom cards', body: (
    <p>A custom card has exactly one day of price data: the fair value the boosted-tree model predicts for it. Your browser builds its
      features from what you declare (expected demand stands in for EDHREC rank; pull rates come from rarity and
      finish defaults), prices it with the boosted-tree model, then feeds that as day 0 to the TCGplayer forecast, which usually answers
      “about no move, wide range”. That is the honest answer with one day of data. There is no buylist forecast,
      and the deep GP covers existing printings only.</p>
  ) },
  { key: 'limits', title: 'Limits', body: (
    <ul className="doc-list">
      <li>Prices cover one summer (late June to early October 2026). Models go stale as the market moves and accuracy decays: BUY hit rates fell from about 77% to 68% over two weeks out of time.</li>
      <li>Expensive cards are hardest: fair-value ranges under-cover $100+ printings.</li>
      <li>Every model leans on EDHREC rank; demand driven by combos, bans or collectibility is mostly invisible.</li>
      <li>Multi-face cards, digital-only and oversized printings are out of scope.</li>
      <li>Not financial advice.</li>
    </ul>
  ) },
  { key: 'build', title: 'How the site is built', body: (
    <ol className="doc-list">
      <li>Nightly, a GitHub Action pulls the newest prices from MTGJSON and card data from Scryfall.</li>
      <li>It retrains Models A, B and C and scores every tradeable printing ({count(67612)} today) with all of them.</li>
      <li>Results are written as static JSON shards, one per set; Models A and B are exported for the browser.</li>
      <li>The site is deployed to GitHub Pages. No server: custom cards are priced on your machine.</li>
    </ol>
  ), note: (
    <>Source: <a href="https://github.com/CXu0630/mtg_price_predict_catboost_deep_gp">github.com/CXu0630/mtg_price_predict_catboost_deep_gp</a>.
      Models and dataset on Hugging Face: <a href="https://huggingface.co/CXu0630/mtg-price-deep-gp">mtg-price-deep-gp</a>,{' '}
      <a href="https://huggingface.co/CXu0630/mtg-price-catboost">mtg-price-catboost</a>,{' '}
      <a href="https://huggingface.co/datasets/CXu0630/mtg-card-market-dataset">mtg-card-market-dataset</a>.</>
  ) },
]

// ---------------------------------------------------------------- Bot trading
const BOTS: Block[] = [
  { key: 'headline', body: (
    <Stats items={[
      { label: 'Best rule, final week', value: '+13.7%', note: 'cash, buylist rule' },
      { label: 'After condition markdown', value: '−9.2%', note: 'CK pays ~12% less below near-mint' },
      { label: 'Random picks', value: '−43.9%', note: 'same prices and costs' },
    ]} />
  ) },
  { key: 'how', title: 'How the bot trades', body: (
    <ul className="doc-list">
      <li>Starts with $100 of fake money and up to 5 positions; at most 4 copies of a printing, each printing once per run.</li>
      <li>Buys at the cheapest real ask, not TCGplayer's market price, and only printings priced $1 or more.</li>
      <li>Sells to Card Kingdom's buylist: $7 shipping per order; a card can't be sold until 4 days after buying; CK pays 7 days after a sale.</li>
      <li>Sells once CK pays 10% over cost, on a SELL signal, or on day 14.</li>
      <li>Walk-forward with no look-ahead: each week is scored only by models trained before it.</li>
      <li>Never trades a CK offer of 1.25× the ask or more: a spot check found those are mostly version mismatches or near-mint premiums, not real offers.</li>
    </ul>
  ) },
  { key: 'rules', title: 'Pick rules', body: (
    <dl className="doc-defs">
      <dt>Buylist</dt><dd>Buy where CK's buylist plus the buylist forecast's pessimistic (10%) quantile already beats the ask.</dd>
      <dt>Forecast</dt><dd>The TCGplayer forecast's BUY signals, best range bottom first.</dd>
      <dt>Gap</dt><dd>The printings furthest below their boosted-tree fair value.</dd>
      <dt>Random</dt><dd>No model at all, 50 runs: the baseline every rule must beat.</dd>
    </dl>
  ), note: 'Forecast and gap picks are bought only when the expected round trip to the exit is positive (the cost gate).' },
  { key: 'final-chart', title: 'Final week (Sep 14–18), cash return per $100', body: (
    <Bars signed format={pct} rows={[
      { label: 'Buylist', value: 13.7, note: '10 buys, 80% winning' },
      { label: 'Forecast', value: -20.0, note: '7 buys, 86% winning' },
      { label: 'Gap', value: -25.8, note: '6 buys, 83% winning' },
      { label: 'Random', value: -43.9, note: 'mean of 50 runs' },
    ]} />
  ), note: 'Forecast and gap picks win most trades and still lose money: two $7 orders are 14% of a $100 bankroll.' },
  { key: 'results', title: 'All results, official rules', body: (
    <Table head={['Run', 'Pick', 'Cash', 'Store credit', 'Cash, −12.2%', 'Buys', 'Winning']}
      align={['l', 'l', 'r', 'r', 'r', 'r', 'r']} rows={[
        ['4 dev weeks', 'buylist', '+17.4%', '+78.2%', '−7.3%', '37', '92%'],
        ['', 'forecast', '−29.9%', '−5.7%', '−39.7%', '22', '59%'],
        ['', 'gap', '−26.2%', '+4.3%', '−38.6%', '29', '76%'],
        ['Final week', 'buylist', '+13.7%', '+69.9%', '−9.2%', '10', '80%'],
        ['', 'forecast', '−20.0%', '+20.1%', '−36.3%', '7', '86%'],
        ['', 'gap', '−25.8%', '−7.6%', '−33.2%', '6', '83%'],
      ]} />
  ), note: 'Store credit is CK sales × 1.30, but it is spent at CK retail, about 23% above TCGplayer, so a credit dollar is worth roughly 1.06 TCGplayer dollars. Selling on TCGplayer (−13% fees) never traded: no forecast cleared the round-trip cost.' },
  { key: 'learned', title: 'What we learned', body: (
    <ul className="doc-list">
      <li>The forecasts rank 14-day moves (rank correlation 0.22–0.24 out of sample), but the edge is a few percent, far below trading costs.</li>
      <li>The only rule that made cash is really “buy where CK's buylist already covers the ask”, and it depends on every copy grading near-mint.</li>
      <li>Uncapped, the bot looked far better (+77% in the final week); all of that came from CK offers that are probably bad data.</li>
      <li>The fair-value gap keeps predicting even at the extremes, but by about +0.5% from the ask: nowhere near the 13% TCGplayer fee.</li>
    </ul>
  ) },
  { key: 'caveats', title: 'Caveats', body: (
    <ul className="doc-list">
      <li>Small evidence: one summer of prices, 4 overlapping development weeks and a single final week with 6–10 trades per rule.</li>
      <li>Fills are assumed: 4 copies at the ask, and CK taking every copy at its posted price.</li>
      <li>CK's own prices can't be verified within its terms; they are judged only against other markets.</li>
    </ul>
  ), note: 'Bot and reports by the teammate project (pwoodsCMU/mtgpricepredictor): paper_trade.py, reports/A3_buylist_bot.md, reports/C_final_test.md.' },
]

// ---------------------------------------------------------------- Data exploration
const t = explore.totals
const DATA: Block[] = [
  { key: 'totals', body: (
    <Stats items={[
      { label: 'Printings × finishes', value: count(t.rows), note: `${count(t.priced)} with a price` },
      { label: 'Distinct cards', value: count(t.cards), note: `across ${count(t.sets)} sets` },
      { label: 'Median price', value: money(t.median), note: `80% between ${money(t.p10)} and ${money(t.p90)}` },
    ]} />
  ), note: `TCGplayer average price over ${t.window}. One row per printing and finish; tokens, digital-only, oversized and multi-face cards excluded.` },
  { key: 'hist', title: 'Most printings are bulk; value is in a long tail', body: (
    <Histogram bins={explore.histogram} markers={[
      { label: 'p10', value: t.p10 }, { label: 'median', value: t.median }, { label: 'p90', value: t.p90 },
    ]} />
  ), note: 'Printings by average price, log scale. Hover a bar for its count.' },
  { key: 'buckets', title: 'Where the money is', body: (
    <Table head={['Price', 'Printings', 'Share of printings', 'Share of total value']} align={['l', 'r', 'r', 'r']}
      rows={explore.buckets.map((b) => [b.label, count(b.n), `${b.share}%`, `${b.value_share}%`])} />
  ), note: 'The 1% of printings at $100+ hold over half the dataset\'s total value; the 61% under $1 hold 2%.' },
  { key: 'rarity', title: 'Median price by rarity', body: (
    <Bars format={money} rows={explore.rarity.map((r) => ({ label: r.label, value: r.median, note: `${count(r.n)} printings` }))} />
  ) },
  { key: 'edhrec', title: 'Demand matters most: median price by EDHREC rank', body: (
    <Bars format={money} rows={explore.edhrec.map((r) => ({ label: r.label, value: r.median }))} />
  ), note: `Cards in the top fifth of Commander play have a median price ${(explore.edhrec[0].median / explore.edhrec[4].median).toFixed(0)}× the bottom fifth's. Reserved List printings (never reprinted) have a median of ${money(explore.reserved.median_reserved)}, against ${money(explore.reserved.median_other)} for everything else.` },
  { key: 'importance', title: 'What drives the fair-value models', body: (
    <PairedBars series={['CatBoost', 'Deep GP']} format={(v) => `+${Math.round(v)}%`}
      rows={explore.importance.map((r) => ({ label: r.label, a: r.catboost ?? undefined, b: r.gp ?? undefined }))} />
  ), note: 'Permutation importance: how much each model\'s held-out error rises when that input is shuffled. Correlated inputs share credit (rarity ↔ pull rates).' },
  { key: 'finish', title: 'Median price by finish', body: (
    <Bars format={money} rows={explore.finish.map((r) => ({ label: r.label, value: r.median, note: `${count(r.n)} printings` }))} />
  ), note: 'Specialty foils are few but command a premium over plain foils, which in turn roughly triple nonfoil.' },
  { key: 'trend', title: 'One summer of movement', body: (
    <Stats items={[
      { label: 'Rose more than 10%', value: `${explore.trend.up10}%` },
      { label: 'Fell more than 10%', value: `${explore.trend.down10}%` },
      { label: 'Median move', value: pct(explore.trend.median_move) },
    ]} />
  ), note: `${count(explore.trend.n)} printings priced $1 or more: mean price in the window's first month vs its last.` },
  { key: 'expensive', title: 'The ten most expensive printings', body: (
    <Table head={['Card', 'Set', 'Finish', 'Average price']} align={['l', 'l', 'l', 'r']}
      rows={explore.expensive.map((e) => [e.name, `${e.set_name} (${e.set.toUpperCase()})`, e.finish, money(e.price)])} />
  ) },
]

export const PAGE_BLOCKS: Record<Exclude<PageId, 'predictor'>, Block[]> = { docs: DOCS, bots: BOTS, data: DATA }
