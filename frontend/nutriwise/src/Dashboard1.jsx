import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import "./Dashboard.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const MEALS_KEY = "nutriwise-meals";

function loadMeals() {
  try {
    return JSON.parse(localStorage.getItem(MEALS_KEY)) || [];
  } catch {
    return [];
  }
}

function saveMeals(meals) {
  try {
    localStorage.setItem(MEALS_KEY, JSON.stringify(meals));
  } catch {
    /* storage full or blocked: ignore */
  }
}

function pretty(name) {
  const s = name.replace(/-/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function formatTime(iso) {
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    hour: "numeric",
    minute: "2-digit",
  });
}

function isToday(iso) {
  return new Date(iso).toDateString() === new Date().toDateString();
}

function makeThumb(file) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const s = 120 / Math.max(img.width, img.height);
      const c = document.createElement("canvas");
      c.width = Math.max(1, Math.round(img.width * s));
      c.height = Math.max(1, Math.round(img.height * s));
      c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
      URL.revokeObjectURL(url);
      resolve(c.toDataURL("image/jpeg", 0.6));
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve("");
    };
    img.src = url;
  });
}

function Dashboard() {
  const [meals, setMeals] = useState(loadMeals);
  const [activeId, setActiveId] = useState(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    saveMeals(meals);
  }, [meals]);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const active = meals.find((m) => m.id === activeId) || null;
  const todayKcal = meals
    .filter((m) => isToday(m.time))
    .reduce((sum, m) => sum + (m.totals?.calories_kcal || 0), 0);

  function startNew() {
    setActiveId(null);
    setFile(null);
    setPreview("");
    setError("");
  }

  function onPick(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Please choose an image file.");
      return;
    }
    setError("");
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${API_URL}/predict`, { method: "POST", body });
      if (!res.ok) {
        const msg = await res.json().catch(() => ({}));
        throw new Error(msg.detail || `Server error (${res.status})`);
      }
      const data = await res.json();
      const thumb = await makeThumb(file);
      const meal = {
        id: Date.now(),
        time: new Date().toISOString(),
        thumb,
        items: data.items,
        totals: data.totals,
        total_grams: data.total_grams,
      };
      setMeals((prev) => [meal, ...prev].slice(0, 30));
      setActiveId(meal.id);
      setFile(null);
      setPreview("");
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Could not reach the server. Make sure the backend is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  function clearAll() {
    if (window.confirm("Delete all saved meals from this device?")) {
      setMeals([]);
      startNew();
    }
  }

  return (
    <div className="dash-page">
      <header className="dash-header">
        <Link to="/" className="logo">
          <span className="logo-mark">
            <span></span>
            <span></span>
            <span></span>
          </span>
          <span>nutriwise</span>
        </Link>
      </header>

      <div className="dash-layout">
        <main className="dash-main">
          {active ? (
            <MealDetail meal={active} onNew={startNew} />
          ) : (
            <>
              <div className="step-heading">
                <div className="step-tag">NEW MEAL</div>
                <h1>Log a meal.</h1>
                <p>Upload a top-down photo of your plate to see its nutrition.</p>
              </div>

              <label className="upload-box">
                <input type="file" accept="image/*" onChange={onPick} hidden />
                {preview ? (
                  <img src={preview} alt="Your plate" className="upload-preview" />
                ) : (
                  <div className="upload-empty">
                    <span className="upload-icon">＋</span>
                    <strong>Choose a photo</strong>
                    <small>JPG or PNG from your device</small>
                  </div>
                )}
              </label>

              {error && <p className="analyze-error">{error}</p>}

              <div className="analyze-actions">
                {preview && (
                  <label className="text-button change-photo">
                    Change photo
                    <input type="file" accept="image/*" onChange={onPick} hidden />
                  </label>
                )}
                <button
                  className="primary-button"
                  disabled={!file || loading}
                  onClick={analyze}
                >
                  {loading ? "Analyzing…" : "Analyze my plate"}
                  <span>→</span>
                </button>
              </div>

              {loading && (
                <p className="analyze-note">This can take 20 to 60 seconds. Hang tight.</p>
              )}
            </>
          )}
        </main>

        <aside className="dash-panel">
          <div className="dash-panel-top">
            <h2>Previous meals</h2>
            <button className="dash-new" onClick={startNew}>
              ＋ New meal
            </button>
          </div>

          <div className="dash-today">
            <span>Today so far</span>
            <strong>{Math.round(todayKcal)} kcal</strong>
          </div>

          {meals.length === 0 ? (
            <p className="dash-empty">No meals yet. Your logged meals will appear here.</p>
          ) : (
            <ul className="dash-list">
              {meals.map((m) => (
                <li key={m.id}>
                  <button
                    className={`dash-meal ${m.id === activeId ? "active" : ""}`}
                    onClick={() => setActiveId(m.id)}
                  >
                    {m.thumb ? (
                      <img src={m.thumb} alt="" />
                    ) : (
                      <span className="dash-thumb-empty">🍽</span>
                    )}
                    <span className="dash-meal-info">
                      <small>{formatTime(m.time)}</small>
                      <strong>
                        {m.totals ? `${Math.round(m.totals.calories_kcal)} kcal` : "—"}
                      </strong>
                      <em>
                        {m.items.slice(0, 3).map((i) => pretty(i.name)).join(", ") ||
                          "No dishes"}
                      </em>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}

          {meals.length > 0 && (
            <button className="dash-clear" onClick={clearAll}>
              Clear history
            </button>
          )}
        </aside>
      </div>
    </div>
  );
}

function MealDetail({ meal, onNew }) {
  const t = meal.totals;
  const stats = t
    ? [
        ["Calories", Math.round(t.calories_kcal), "kcal"],
        ["Protein", t.protein_g.toFixed(1), "g"],
        ["Carbs", t.carbs_g.toFixed(1), "g"],
        ["Fat", t.fat_g.toFixed(1), "g"],
        ["Fiber", t.fiber_g.toFixed(1), "g"],
      ]
    : [];

  return (
    <>
      <div className="step-heading">
        <div className="step-tag">{formatTime(meal.time).toUpperCase()}</div>
        <h1>Meal summary.</h1>
        <p>About {Math.round(meal.total_grams)} g of food on this plate.</p>
      </div>

      {t ? (
        <div className="dash-stats">
          {stats.map(([label, value, unit]) => (
            <div key={label} className={`dash-stat ${label === "Calories" ? "big" : ""}`}>
              <span>{label}</span>
              <strong>{value}</strong>
              <small>{unit}</small>
            </div>
          ))}
        </div>
      ) : (
        <p className="analyze-note">No nutrition data found for this plate.</p>
      )}

      {meal.items.length > 0 && (
        <div className="dash-items">
          <h3>What we found</h3>
          {meal.items.map((it) => (
            <div key={it.id} className="dash-item">
              <span>{pretty(it.name)}</span>
              <span>{it.weight_g != null ? `${Math.round(it.weight_g)} g` : "—"}</span>
              <span>
                {it.nutrition ? `${Math.round(it.nutrition.calories_kcal)} kcal` : "—"}
              </span>
            </div>
          ))}
        </div>
      )}

      <p className="analyze-note">
        Portion sizes and nutrition are estimates, not exact measurements.
      </p>

      <div className="analyze-actions">
        <button className="primary-button" onClick={onNew}>
          Log another meal
          <span>＋</span>
        </button>
      </div>
    </>
  );
}

export default Dashboard;