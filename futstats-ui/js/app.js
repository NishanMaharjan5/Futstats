/* ==========================================================
   FutStats — app.js
   All rendering is driven off a single data object shaped
   like results.json. See js/data.js for the schema + sample.
   ========================================================== */

(function () {
  "use strict";

  let DATA = null;
  let sortState = { key: "rating", dir: "desc" };
  let teamFilter = "all";

  /* ---------------- data loading ---------------- */

  function loadData() {
    return fetch("data/results.json")
      .then((res) => {
        if (!res.ok) throw new Error("no results.json found");
        return res.json();
      })
      .catch(() => window.FUTSTATS_DATA);
  }

  /* ---------------- helpers ---------------- */

  function el(tag, cls, html) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (html !== undefined) node.innerHTML = html;
    return node;
  }

  function ratingTier(rating) {
    if (rating >= 7) return "rating-good";
    if (rating >= 5.5) return "rating-mid";
    return "rating-low";
  }

  function teamName(team) {
    return team === "A" ? DATA.match_info.team_a_name : DATA.match_info.team_b_name;
  }

  /**
   * Renders an image into a frame container, falling back to a quiet
   * "not generated yet" placeholder if the path is empty or fails to load.
   * tagLabel/tagWarn optionally add a small pill in the top-left corner.
   */
  function renderImageFrame(container, src, emptyText, tagLabel, tagWarn) {
    container.innerHTML = "";
    if (tagLabel) {
      const tag = el("div", "frame-tag" + (tagWarn ? " frame-tag-warn" : ""), tagLabel);
      container.appendChild(tag);
    }
    if (!src) {
      renderEmptyFrame(container, emptyText);
      return;
    }
    const img = new Image();
    img.alt = emptyText || "";
    img.onerror = () => renderEmptyFrame(container, emptyText);
    img.onload = () => container.appendChild(img);
    img.src = src;
  }

  function renderEmptyFrame(container, text) {
    // clear anything but tag pills
    const tag = container.querySelector(".frame-tag");
    container.innerHTML = "";
    if (tag) container.appendChild(tag);
    const wrap = el("div", "img-frame-empty");
    wrap.appendChild(el("div", "pitch-icon"));
    wrap.appendChild(el("span", null, text || "Image not generated yet"));
    container.appendChild(wrap);
  }

  /* ---------------- section renderers ---------------- */

  function renderMatchbar() {
    const bar = document.getElementById("matchbar");
    bar.innerHTML = "";

    const a = el("div", "side");
    a.appendChild(el("span", "team-dot team-a"));
    a.appendChild(document.createTextNode(DATA.match_info.team_a_name));
    bar.appendChild(a);

    const meta = el("div", "meta", `
      <span>${DATA.match_info.venue}</span>
      <span>${DATA.match_info.date} · ${DATA.match_info.duration_min} min</span>
    `);
    bar.appendChild(meta);

    const b = el("div", "side");
    b.appendChild(document.createTextNode(DATA.match_info.team_b_name));
    b.appendChild(el("span", "team-dot team-b"));
    bar.appendChild(b);
  }

  function renderPossession(targetId) {
    const wrap = document.getElementById(targetId);
    const { team_a_pct, team_b_pct } = DATA.possession;
    wrap.innerHTML = "";

    const outer = el("div", "poss-bar-wrap");

    const labels = el("div", "poss-labels");
    const left = el("div", "team-name");
    left.innerHTML = `<span class="team-dot team-a"></span>${DATA.match_info.team_a_name}`;
    const pctA = el("span", "pct", team_a_pct + "%");
    pctA.style.color = "var(--team-a)";
    const pctB = el("span", "pct", team_b_pct + "%");
    pctB.style.color = "var(--team-b)";
    const right = el("div", "team-name");
    right.innerHTML = `${DATA.match_info.team_b_name}<span class="team-dot team-b" style="margin-left:7px;margin-right:0;"></span>`;

    const labelsTop = el("div", "poss-labels");
    labelsTop.appendChild(pctA);
    labelsTop.appendChild(pctB);
    outer.appendChild(labelsTop);

    const track = el("div", "poss-track");
    const segA = el("div", "poss-seg-a");
    segA.style.width = team_a_pct + "%";
    const segB = el("div", "poss-seg-b");
    segB.style.width = team_b_pct + "%";
    track.appendChild(segA);
    track.appendChild(segB);
    outer.appendChild(track);

    const namesRow = el("div", "poss-labels");
    namesRow.appendChild(left);
    namesRow.appendChild(right);
    outer.appendChild(namesRow);

    wrap.appendChild(outer);
  }

  function renderOverview() {
    renderPossession("possession-bar-hero");

    const mapSummary = document.getElementById("map-summary");
    mapSummary.innerHTML = `
      <span class="big-num">${DATA.model_info.map50}%</span>
      <span class="big-unit">mAP@50 on our labeled test set</span>
    `;

    renderImageFrame(
      document.getElementById("overview-heatmap"),
      DATA.assets.heatmap_match,
      "Match heatmap renders here once available"
    );

    const topPlayers = [...DATA.players]
      .sort((a, b) => b.rating - a.rating)
      .slice(0, 5);

    const list = document.getElementById("top-players");
    list.innerHTML = "";
    topPlayers.forEach((p) => {
      const row = el("div", "top-player-row");
      row.innerHTML = `
        <span><span class="team-dot team-${p.team.toLowerCase()}"></span><span class="pid">#${p.id}</span> — ${teamName(p.team)}</span>
        <span class="rating-badge ${ratingTier(p.rating)}">${p.rating.toFixed(1)}</span>
      `;
      list.appendChild(row);
    });
  }

  function renderHeatmaps() {
    renderImageFrame(document.getElementById("heatmap-match"), DATA.assets.heatmap_match, "Match heatmap renders here once available");
    renderImageFrame(document.getElementById("heatmap-team-a"), DATA.assets.heatmap_team_a, "Team heatmap renders here once available");
    renderImageFrame(document.getElementById("heatmap-team-b"), DATA.assets.heatmap_team_b, "Team heatmap renders here once available");
    document.getElementById("hm-team-a-name").textContent = DATA.match_info.team_a_name;
    document.getElementById("hm-team-b-name").textContent = DATA.match_info.team_b_name;
  }

  function renderPassNetworks() {
    renderImageFrame(document.getElementById("passnet-team-a"), DATA.assets.pass_network_team_a, "Pass network renders here once available");
    renderImageFrame(document.getElementById("passnet-team-b"), DATA.assets.pass_network_team_b, "Pass network renders here once available");
    document.getElementById("pn-team-a-name").textContent = DATA.match_info.team_a_name;
    document.getElementById("pn-team-b-name").textContent = DATA.match_info.team_b_name;
  }

  function renderLimitations() {
    renderImageFrame(document.getElementById("radar-before"), DATA.assets.radar_before, "Reference broadcast-style radar view", "Expected");
    renderImageFrame(document.getElementById("radar-after"), DATA.assets.radar_after, "Drifted output from our unstabilised camera", "What we actually got", true);
  }

  function renderAbout() {
    const stats = document.getElementById("about-stats");
    stats.innerHTML = "";
    const items = [
      { num: DATA.model_info.map50 + "%", label: "mAP@50 detection accuracy" },
      { num: DATA.model_info.frames_labeled.toLocaleString(), label: "Frames hand-labeled" },
      { num: DATA.model_info.annotations.toLocaleString(), label: "Total annotations" }
    ];
    items.forEach((it) => {
      const block = el("div", "stat-block");
      block.innerHTML = `<span class="stat-num">${it.num}</span><span class="stat-label">${it.label}</span>`;
      stats.appendChild(block);
    });
    document.getElementById("about-dataset-note").textContent = DATA.model_info.dataset_note;
  }

  function renderVideo() {
    const videoEl = document.getElementById("video-el");
    const placeholder = document.getElementById("video-placeholder");
    document.getElementById("v-team-a-name").textContent = DATA.match_info.team_a_name;
    document.getElementById("v-team-b-name").textContent = DATA.match_info.team_b_name;

    if (DATA.assets.video) {
      videoEl.src = DATA.assets.video;
      videoEl.style.display = "block";
      placeholder.style.display = "none";
    } else {
      videoEl.style.display = "none";
      placeholder.style.display = "flex";
    }
  }

  /* ---------------- players table ---------------- */

  function renderTeamFilter() {
    const wrap = document.getElementById("team-filter");
    wrap.innerHTML = "";
    const options = [
      { key: "all", label: "All" },
      { key: "A", label: DATA.match_info.team_a_name },
      { key: "B", label: DATA.match_info.team_b_name }
    ];
    options.forEach((opt) => {
      const chip = el("button", "chip" + (teamFilter === opt.key ? " is-active" : ""), opt.label);
      chip.dataset.team = opt.key;
      chip.addEventListener("click", () => {
        teamFilter = opt.key;
        renderTeamFilter();
        renderPlayerTable();
      });
      wrap.appendChild(chip);
    });
  }

  function renderPlayerTable() {
    const body = document.getElementById("player-table-body");
    body.innerHTML = "";

    let rows = DATA.players.filter((p) => teamFilter === "all" || p.team === teamFilter);

    rows.sort((a, b) => {
      let va = a[sortState.key];
      let vb = b[sortState.key];
      if (sortState.key === "id") { va = String(va); vb = String(vb); }
      if (typeof va === "string") {
        return sortState.dir === "asc" ? va.localeCompare(vb) : vb.localeCompare(va);
      }
      return sortState.dir === "asc" ? va - vb : vb - va;
    });

    rows.forEach((p) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="pid-cell">#${p.id}</td>
        <td class="team-cell"><span class="team-dot team-${p.team.toLowerCase()}"></span>${teamName(p.team)}</td>
        <td>${p.touches}</td>
        <td>${p.passes}</td>
        <td>${p.turnovers}</td>
        <td>#${p.activity_rank}</td>
        <td><span class="rating-badge ${ratingTier(p.rating)}">${p.rating.toFixed(1)}</span></td>
      `;
      body.appendChild(tr);
    });

    // update sorted header highlight
    document.querySelectorAll("#player-table thead th").forEach((th) => {
      th.classList.toggle("sorted", th.dataset.sort === sortState.key);
    });
  }

  function bindTableSort() {
    document.querySelectorAll("#player-table thead th").forEach((th) => {
      th.addEventListener("click", () => {
        const key = th.dataset.sort;
        const keyMap = { rating: "rating", touches: "touches", passes: "passes", turnovers: "turnovers", activity: "activity_rank", id: "id", team: "team" };
        const mapped = keyMap[key] || key;
        if (sortState.key === mapped) {
          sortState.dir = sortState.dir === "asc" ? "desc" : "asc";
        } else {
          sortState.key = mapped;
          sortState.dir = mapped === "activity_rank" ? "asc" : "desc";
        }
        renderPlayerTable();
      });
    });
  }

  /* ---------------- tab routing ---------------- */

  function showTab(tabKey) {
    document.querySelectorAll(".nav-item").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.tab === tabKey);
    });
    document.querySelectorAll(".tab").forEach((section) => {
      section.classList.toggle("is-active", section.id === "tab-" + tabKey);
    });
  }

  function bindNav() {
    document.querySelectorAll(".nav-item").forEach((btn) => {
      btn.addEventListener("click", () => showTab(btn.dataset.tab));
    });
    document.querySelectorAll("[data-goto]").forEach((btn) => {
      btn.addEventListener("click", () => showTab(btn.dataset.goto));
    });
  }

  /* ---------------- theme switcher ---------------- */

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    const label = document.querySelector("#theme-toggle .theme-label");
    if (label) {
      label.textContent = theme === "light" ? "Light Mode" : "Dark Mode";
    }
    try {
      localStorage.setItem("futstats-theme", theme);
    } catch (e) {}
  }

  function initTheme() {
    let saved = "dark";
    try {
      saved = localStorage.getItem("futstats-theme") || "dark";
    } catch (e) {}
    applyTheme(saved);

    const toggleBtn = document.getElementById("theme-toggle");
    if (toggleBtn) {
      toggleBtn.addEventListener("click", () => {
        const current = document.documentElement.dataset.theme === "light" ? "light" : "dark";
        const next = current === "dark" ? "light" : "dark";
        applyTheme(next);
      });
    }
  }

  /* ---------------- init ---------------- */

  initTheme();

  loadData().then((data) => {
    DATA = data;
    renderMatchbar();
    renderOverview();
    renderHeatmaps();
    renderPassNetworks();
    renderTeamFilter();
    renderPlayerTable();
    bindTableSort();
    renderVideo();
    renderLimitations();
    renderAbout();
    bindNav();
  });
})();
