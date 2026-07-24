/* FutStats dashboard — all stats come from data/results.json (nothing hardcoded). */

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
};

const pad2 = n => String(n).padStart(2, "0");
const fmtDuration = s => `${pad2(Math.floor(s / 60))}:${pad2(s % 60)}`;

function teamColorMap(info) {
  const m = {};
  info.teams.forEach((t, i) => (m[t] = info.team_colors[i]));
  return m;
}

// A kit color this dark needs a light ring to stay visible on the dark surface.
function isDark(hex) {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16),
    g = parseInt(h.slice(2, 4), 16),
    b = parseInt(h.slice(4, 6), 16);
  return 0.299 * r + 0.587 * g + 0.114 * b < 70;
}

function dot(color, small) {
  const d = el("span", "dot" + (small ? " sm" : ""));
  d.style.background = color;
  if (isDark(color)) d.classList.add("dark-kit");
  return d;
}

// Rating badge is a STATUS color that always carries the number (never color-alone).
const ratingClass = r => (r >= 7 ? "good" : r >= 6 ? "warn" : "bad");

const STATE = { data: null, sortKey: "rating", sortDir: -1 };

fetch("data/results.json")
  .then(r => {
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  })
  .then(data => {
    STATE.data = data;
    render(data);
  })
  .catch(showError);

function showError(err) {
  const b = $("#error");
  b.hidden = false;
  b.innerHTML =
    `<strong>Couldn't load data/results.json</strong> — ${err.message}. ` +
    `If you opened this page directly (file://), the browser blocks <code>fetch()</code>. ` +
    `Run <code>python3 -m http.server 8000</code> in the FutStats folder, then open ` +
    `<a href="http://localhost:8000">http://localhost:8000</a>.`;
}

function render(d) {
  renderMeta(d.match_info);
  renderVideo(d.assets);
  renderPossession(d);
  renderTiles(d);
  wireSorting();
  renderTable(d);
  renderVisuals(d.assets, d.match_info);
}

function renderMeta(info) {
  $("#match-teams").textContent = `${info.teams[0]} vs ${info.teams[1]}`;
  const bits = [fmtDuration(info.duration_sec)];
  if (info.venue) bits.push(info.venue);
  if (info.date) bits.push(info.date);
  $("#match-sub").textContent = bits.join(" · ");
}

function renderVideo(assets) {
  const host = $("#video-media");
  host.innerHTML = "";
  const v = el("video");
  v.controls = true;
  v.preload = "metadata";
  v.poster = "assets/video_placeholder.png"; // shown until a real clip is added
  const src = el("source");
  src.src = assets.video;
  src.type = "video/mp4";
  v.appendChild(src);
  host.appendChild(v);
}

function renderPossession(d) {
  const { teams, team_colors } = d.match_info;
  const p = d.possession;
  const host = $("#possession");
  host.innerHTML = "";
  host.appendChild(el("div", "card-title", "Possession"));

  const body = el("div", "poss-body");
  const head = el("div", "poss-head");
  head.appendChild(side(teams[0], team_colors[0], p.team_a_pct, "left"));
  head.appendChild(side(teams[1], team_colors[1], p.team_b_pct, "right"));
  body.appendChild(head);

  const bar = el("div", "poss-bar");
  bar.appendChild(seg(team_colors[0], p.team_a_pct));
  bar.appendChild(seg(team_colors[1], p.team_b_pct));
  body.appendChild(bar);
  host.appendChild(body);

  function side(name, color, pct, dir) {
    const s = el("div", "poss-side " + dir);
    const nm = el("span", "poss-name", name);
    const val = el("span", "poss-pct", pct.toFixed(1) + "%");
    if (dir === "left") s.append(dot(color), nm, val);
    else s.append(val, nm, dot(color));
    return s;
  }
  function seg(color, pct) {
    const s = el("div", "seg");
    s.style.width = pct + "%";
    s.style.background = color;
    if (isDark(color)) s.classList.add("dark-kit");
    return s;
  }
}

function renderTiles(d) {
  const t = d.totals;
  const host = $("#tiles");
  host.innerHTML = "";
  const mvp = d.players.find(p => p.id === t.most_active_player_id);

  host.appendChild(tile(t.touches, "Total touches"));
  host.appendChild(tile(t.passes, "Total passes"));
  host.appendChild(tile(t.turnovers, "Total turnovers"));
  host.appendChild(
    tile(
      mvp ? "#" + mvp.id : "—",
      "Most active player",
      true,
      mvp ? `${mvp.team} · ${mvp.touches} touches` : ""
    )
  );

  function tile(value, label, accent, sub) {
    const c = el("div", "card tile");
    c.appendChild(el("div", "tile-value" + (accent ? " accent" : ""), value));
    c.appendChild(el("div", "label", label));
    if (sub) c.appendChild(el("div", "tile-sub", sub));
    return c;
  }
}

function wireSorting() {
  document.querySelectorAll("#players th.sortable").forEach(th => {
    if (th.dataset.wired) return;
    th.dataset.wired = "1";
    th.dataset.label = th.textContent;
    th.addEventListener("click", () => {
      const k = th.dataset.key;
      if (STATE.sortKey === k) STATE.sortDir *= -1;
      else {
        STATE.sortKey = k;
        STATE.sortDir = k === "team" ? 1 : -1; // numbers high→low, text A→Z
      }
      renderTable(STATE.data);
    });
  });
}

function renderTable(d) {
  const cmap = teamColorMap(d.match_info);
  const tbody = $("#players tbody");
  tbody.innerHTML = "";

  const rows = [...d.players].sort((a, b) => {
    const x = a[STATE.sortKey],
      y = b[STATE.sortKey];
    return STATE.sortDir * (typeof x === "string" ? x.localeCompare(y) : x - y);
  });

  for (const p of rows) {
    const tr = el("tr");
    if (p.id === d.totals.most_active_player_id) tr.classList.add("mvp");

    tr.appendChild(el("td", "mono", "#" + p.id));

    const team = el("td");
    team.append(dot(cmap[p.team], true), document.createTextNode(" " + p.team));
    tr.appendChild(team);

    tr.appendChild(el("td", "num", p.touches));
    tr.appendChild(el("td", "num", p.passes));
    tr.appendChild(el("td", "num", p.turnovers));

    const rt = el("td", "num");
    rt.appendChild(el("span", "badge " + ratingClass(p.rating), p.rating.toFixed(1)));
    tr.appendChild(rt);

    tbody.appendChild(tr);
  }
  updateSortIndicators();
}

function updateSortIndicators() {
  document.querySelectorAll("#players th.sortable").forEach(th => {
    const active = th.dataset.key === STATE.sortKey;
    const caret = active ? (STATE.sortDir < 0 ? " ▾" : " ▴") : "";
    th.innerHTML = th.dataset.label + `<span class="caret">${caret}</span>`;
    th.classList.toggle("active", active);
  });
}

function renderVisuals(a, info) {
  const host = $("#visuals");
  host.innerHTML = "";
  host.appendChild(viz("Match heatmap", a.heatmap_match, true));
  host.appendChild(viz(`${info.teams[0]} — heatmap`, a.heatmap_team_a));
  host.appendChild(viz(`${info.teams[1]} — heatmap`, a.heatmap_team_b));
  host.appendChild(viz("Pass network", a.pass_network, true));
  host.appendChild(viz("Event timeline", a.timeline, true));

  function viz(title, src, full) {
    const c = el("div", "card viz" + (full ? " full" : ""));
    c.appendChild(el("div", "card-title", title));
    const img = el("img");
    img.src = src;
    img.alt = title;
    img.loading = "lazy";
    img.addEventListener("error", () =>
      img.replaceWith(el("div", "img-fallback", "Image not found: " + src))
    );
    c.appendChild(img);
    return c;
  }
}
