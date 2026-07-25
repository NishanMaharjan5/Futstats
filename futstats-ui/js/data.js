/**
 * FutStats — sample data
 *
 * This mirrors the exact shape your backend's results.json should produce.
 * The UI reads ONLY from this object (window.FUTSTATS_DATA) via app.js.
 *
 * To wire up real data:
 *   1. Have your pipeline write a results.json in this shape to /data/results.json
 *   2. In app.js, loadData() already tries to fetch('data/results.json') first
 *      and falls back to this sample if that fetch fails (e.g. opening
 *      index.html directly via file:// with no local server).
 *   3. Point `assets` paths at wherever your generated images/video actually live.
 */

window.FUTSTATS_DATA = {
  match_info: {
    team_a_name: "Himal FC",
    team_b_name: "Rani Futsal Club",
    date: "2026-06-14",
    venue: "Sifal Futsal Arena, Kathmandu",
    duration_min: 40
  },

  possession: {
    team_a_pct: 57,
    team_b_pct: 43
  },

  team_balance_check: {
    team_a_players_detected: 7,
    team_b_players_detected: 7,
    note: "internal QA — not surfaced in UI"
  },

  players: [
    { id: "04", team: "A", touches: 212, passes: 61, turnovers: 6, activity_rank: 2, rating: 8.1 },
    { id: "07", team: "A", touches: 176, passes: 48, turnovers: 9, activity_rank: 4, rating: 7.2 },
    { id: "11", team: "A", touches: 268, passes: 74, turnovers: 5, activity_rank: 1, rating: 8.6 },
    { id: "17", team: "A", touches: 84,  passes: 19, turnovers: 4, activity_rank: 9,  rating: 6.4 },
    { id: "17b", team: "A", touches: 61, passes: 14, turnovers: 3, activity_rank: 11, rating: 6.1 },
    { id: "23", team: "A", touches: 143, passes: 33, turnovers: 11, activity_rank: 6, rating: 5.8 },
    { id: "29", team: "A", touches: 98,  passes: 22, turnovers: 7, activity_rank: 8, rating: 5.9 },

    { id: "02", team: "B", touches: 121, passes: 29, turnovers: 8,  activity_rank: 7,  rating: 5.6 },
    { id: "09", team: "B", touches: 187, passes: 52, turnovers: 6,  activity_rank: 3,  rating: 7.4 },
    { id: "13", team: "B", touches: 154, passes: 38, turnovers: 12, activity_rank: 5,  rating: 5.2 },
    { id: "16", team: "B", touches: 71,  passes: 15, turnovers: 5,  activity_rank: 10, rating: 5.9 },
    { id: "21", team: "B", touches: 203, passes: 55, turnovers: 9,  activity_rank: 2,  rating: 6.8 },
    { id: "26", team: "B", touches: 58,  passes: 11, turnovers: 6,  activity_rank: 12, rating: 4.6 },
    { id: "31", team: "B", touches: 132, passes: 27, turnovers: 10, activity_rank: 6,  rating: 5.1 }
  ],

  assets: {
    video: "",
    heatmap_match: "",
    heatmap_team_a: "",
    heatmap_team_b: "",
    pass_network_team_a: "",
    pass_network_team_b: "",
    radar_before: "",
    radar_after: ""
  },

  model_info: {
    map50: 91.4,
    frames_labeled: 18400,
    annotations: 142300,
    dataset_note: "Filmed at local futsal venues around Kathmandu and hand-labeled frame by frame for player and ball position, since no existing model had been trained on this style of footage before. Frames span different lighting, camera angles, and court surfaces to keep the model from overfitting to a single venue."
  }
};
