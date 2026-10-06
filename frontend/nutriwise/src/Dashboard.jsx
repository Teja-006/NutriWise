import React, { useEffect, useMemo, useRef, useState } from "react";
import "./Dashboard.css";

const API_URL =
  import.meta.env.VITE_API_URL ||
  "https://nutriwise-backend-qb2i.onrender.com";

const MEAL_TYPES = [
  { key: "breakfast", label: "Breakfast", number: "01" },
  { key: "lunch", label: "Lunch", number: "02" },
  { key: "dinner", label: "Dinner", number: "03" },
];

const EMPTY_TOTALS = {
  calories: 0,
  protein: 0,
  carbs: 0,
  fat: 0,
  fiber: 0,
};

function getTodayKey() {
  const date = new Date();

  return [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, "0"),
    String(date.getDate()).padStart(2, "0"),
  ].join("-");
}

function formatDateLong(dateKey) {
  const [year, month, day] = dateKey.split("-").map(Number);

  return new Date(year, month - 1, day).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "long",
    year: "numeric",
  });
}

function formatDateShort(dateKey) {
  const [year, month, day] = dateKey.split("-").map(Number);

  return new Date(year, month - 1, day).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function createEmptyDay() {
  return {
    breakfast: [],
    lunch: [],
    dinner: [],
  };
}

function loadHistory() {
  try {
    const saved = localStorage.getItem("nutriwise-meal-history");
    return saved ? JSON.parse(saved) : {};
  } catch {
    return {};
  }
}

function roundNumber(value) {
  return Math.round((Number(value) || 0) * 10) / 10;
}

/*
 * ============================================================
 * NUTRITION CALCULATION
 * ============================================================
 *
 * Your backend nutrition.py uses:
 *
 * calories_kcal
 * protein_g
 * carbs_g
 * fat_g
 * fiber_g
 *
 * The frontend also accepts the older names as fallbacks.
 */

function calculateMealNutrition(meal) {
  const totals = {
    calories: 0,
    protein: 0,
    carbs: 0,
    fat: 0,
    fiber: 0,
  };

  const items = meal?.items || [];

  let foundItemNutrition = false;

  items.forEach((item) => {
    const nutrition = item?.nutrition;

    if (!nutrition) return;

    foundItemNutrition = true;

    totals.calories +=
      Number(
        nutrition.calories_kcal ??
          nutrition.calories ??
          nutrition.kcal ??
          0
      ) || 0;

    totals.protein +=
      Number(
        nutrition.protein_g ??
          nutrition.protein ??
          0
      ) || 0;

    totals.carbs +=
      Number(
        nutrition.carbs_g ??
          nutrition.carbs ??
          0
      ) || 0;

    totals.fat +=
      Number(
        nutrition.fat_g ??
          nutrition.fat ??
          0
      ) || 0;

    totals.fiber +=
      Number(
        nutrition.fiber_g ??
          nutrition.fiber ??
          0
      ) || 0;
  });

  /*
   * If individual food nutrition isn't available,
   * use the combined meal nutrition returned by the API.
   */
  if (!foundItemNutrition && meal?.nutrition) {
    const nutrition = meal.nutrition;

    totals.calories =
      Number(
        nutrition.calories_kcal ??
          nutrition.calories ??
          nutrition.kcal ??
          0
      ) || 0;

    totals.protein =
      Number(
        nutrition.protein_g ??
          nutrition.protein ??
          0
      ) || 0;

    totals.carbs =
      Number(
        nutrition.carbs_g ??
          nutrition.carbs ??
          0
      ) || 0;

    totals.fat =
      Number(
        nutrition.fat_g ??
          nutrition.fat ??
          0
      ) || 0;

    totals.fiber =
      Number(
        nutrition.fiber_g ??
          nutrition.fiber ??
          0
      ) || 0;
  }

  return {
    calories: roundNumber(totals.calories),
    protein: roundNumber(totals.protein),
    carbs: roundNumber(totals.carbs),
    fat: roundNumber(totals.fat),
    fiber: roundNumber(totals.fiber),
  };
}

function calculateDayTotals(day) {
  const totals = { ...EMPTY_TOTALS };

  if (!day) return totals;

  MEAL_TYPES.forEach(({ key }) => {
    const meals = day[key] || [];

    meals.forEach((meal) => {
      const nutrition = calculateMealNutrition(meal);

      totals.calories += nutrition.calories;
      totals.protein += nutrition.protein;
      totals.carbs += nutrition.carbs;
      totals.fat += nutrition.fat;
      totals.fiber += nutrition.fiber;
    });
  });

  return {
    calories: roundNumber(totals.calories),
    protein: roundNumber(totals.protein),
    carbs: roundNumber(totals.carbs),
    fat: roundNumber(totals.fat),
    fiber: roundNumber(totals.fiber),
  };
}

function calculateDayWeight(day) {
  if (!day) return 0;

  let total = 0;

  MEAL_TYPES.forEach(({ key }) => {
    (day[key] || []).forEach((meal) => {
      if (meal.total_grams !== undefined) {
        total += Number(meal.total_grams) || 0;
      } else {
        (meal.items || []).forEach((item) => {
          total += Number(item.weight_g) || 0;
        });
      }
    });
  });

  return roundNumber(total);
}

function getMealCount(day) {
  if (!day) return 0;

  return (
    (day.breakfast?.length || 0) +
    (day.lunch?.length || 0) +
    (day.dinner?.length || 0)
  );
}

/* ============================================================
   MEAL CARD
   ============================================================ */

function MealCard({ meal, onDelete }) {
  const nutrition = calculateMealNutrition(meal);

  const weight =
    meal.total_grams !== undefined
      ? Number(meal.total_grams) || 0
      : (meal.items || []).reduce(
          (sum, item) => sum + (Number(item.weight_g) || 0),
          0
        );

  return (
    <div className="logged-meal">
      <div className="logged-meal-top">
        <div>
          <div className="logged-meal-title">
            Meal from picture
          </div>

          <div className="logged-meal-weight">
            {roundNumber(weight)} g
          </div>
        </div>

        <div className="logged-meal-right">
          <div className="logged-meal-calories">
            {nutrition.calories} kcal
          </div>

          <button
            className="delete-meal"
            type="button"
            onClick={() => onDelete(meal.id)}
          >
            ×
          </button>
        </div>
      </div>

      <div className="food-tags">
        {(meal.items || []).map((item, index) => (
          <span
            className="food-tag"
            key={`${item.name}-${index}`}
          >
            {item.name}

            {item.weight_g !== null &&
            item.weight_g !== undefined
              ? ` · ${roundNumber(item.weight_g)} g`
              : ""}
          </span>
        ))}
      </div>

      <div className="meal-macro-row">
        <span>P {nutrition.protein}g</span>
        <span>C {nutrition.carbs}g</span>
        <span>F {nutrition.fat}g</span>
        <span>Fi {nutrition.fiber}g</span>
      </div>
    </div>
  );
}

/* ============================================================
   EMPTY PICTURE BOX
   ============================================================ */

function EmptyMealBox({ onPicture }) {
  return (
    <button
      className="picture-upload-box"
      type="button"
      onClick={onPicture}
    >
      <div className="picture-icon">▣</div>

      <div className="picture-upload-content">
        <strong>Add from picture</strong>
        <span>Upload a picture of your meal</span>
      </div>

      <div className="picture-arrow">+</div>
    </button>
  );
}

/* ============================================================
   DASHBOARD
   ============================================================ */

export default function Dashboard() {
  const today = getTodayKey();

  const [selectedDate, setSelectedDate] = useState(today);
  const [history, setHistory] = useState(loadHistory);
  const [uploadingMealType, setUploadingMealType] =
    useState(null);
  const [error, setError] = useState("");

  const fileInputRef = useRef(null);
  const activeMealTypeRef = useRef(null);

  const currentDay =
    history[selectedDate] || createEmptyDay();

  const totals = useMemo(
    () => calculateDayTotals(currentDay),
    [currentDay]
  );

  const totalFoodWeight = useMemo(
    () => calculateDayWeight(currentDay),
    [currentDay]
  );

  /*
   * Show all dates that actually contain meals.
   */
  const historyDates = useMemo(() => {
    return Object.keys(history)
      .filter((date) => {
        return getMealCount(history[date]) > 0;
      })
      .sort((a, b) => b.localeCompare(a));
  }, [history]);

  useEffect(() => {
    localStorage.setItem(
      "nutriwise-meal-history",
      JSON.stringify(history)
    );
  }, [history]);

  /* ============================================================
     IMAGE UPLOAD
     ============================================================ */

  function openPicturePicker(mealType) {
    setError("");

    activeMealTypeRef.current = mealType;

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
      fileInputRef.current.click();
    }
  }

  async function handlePictureSelected(event) {
    const file = event.target.files?.[0];

    if (!file) return;

    const mealType = activeMealTypeRef.current;

    if (!mealType) return;

    setUploadingMealType(mealType);
    setError("");

    try {
      const formData = new FormData();

      formData.append("file", file);

      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        let message =
          "Could not analyse the image.";

        try {
          const data = await response.json();

          if (data?.detail) {
            message = data.detail;
          }
        } catch {
          // Ignore JSON parsing error.
        }

        throw new Error(message);
      }

      const result = await response.json();

      if (!Array.isArray(result.items)) {
        throw new Error(
          "The API returned an invalid response."
        );
      }

      if (result.items.length === 0) {
        throw new Error(
          "No food was detected. Please try another picture."
        );
      }

      /*
       * ========================================================
       * ONE PICTURE = ONE MEAL
       * ========================================================
       *
       * All detected foods remain inside the same meal.
       */

      const meal = {
        id: `${Date.now()}-${Math.random()
          .toString(36)
          .slice(2)}`,

        source: "picture",

        createdAt: new Date().toISOString(),

        total_grams:
          Number(result.total_grams) || 0,

        items: result.items.map((item) => ({
          id: Number(item.id) || 0,

          name: item.name || "Unknown food",

          confidence:
            item.confidence == null
              ? null
              : Number(item.confidence),

          area_cm2:
            item.area_cm2 == null
              ? null
              : Number(item.area_cm2),

          weight_g:
            item.weight_g == null
              ? null
              : Number(item.weight_g),

          /*
           * IMPORTANT:
           *
           * Your Python nutrition database returns:
           *
           * calories_kcal
           * protein_g
           * carbs_g
           * fat_g
           * fiber_g
           */
          nutrition: item.nutrition
            ? {
                calories_kcal:
                  Number(
                    item.nutrition.calories_kcal ??
                      item.nutrition.calories ??
                      item.nutrition.kcal ??
                      0
                  ) || 0,

                protein_g:
                  Number(
                    item.nutrition.protein_g ??
                      item.nutrition.protein ??
                      0
                  ) || 0,

                carbs_g:
                  Number(
                    item.nutrition.carbs_g ??
                      item.nutrition.carbs ??
                      0
                  ) || 0,

                fat_g:
                  Number(
                    item.nutrition.fat_g ??
                      item.nutrition.fat ??
                      0
                  ) || 0,

                fiber_g:
                  Number(
                    item.nutrition.fiber_g ??
                      item.nutrition.fiber ??
                      0
                  ) || 0,
              }
            : null,
        })),

        /*
         * Keep API totals as a fallback.
         */
        nutrition: {
          calories_kcal:
            Number(
              result.totals?.calories_kcal ??
                result.totals?.calories ??
                result.totals?.kcal ??
                0
            ) || 0,

          protein_g:
            Number(
              result.totals?.protein_g ??
                result.totals?.protein ??
                0
            ) || 0,

          carbs_g:
            Number(
              result.totals?.carbs_g ??
                result.totals?.carbs ??
                0
            ) || 0,

          fat_g:
            Number(
              result.totals?.fat_g ??
                result.totals?.fat ??
                0
            ) || 0,

          fiber_g:
            Number(
              result.totals?.fiber_g ??
                result.totals?.fiber ??
                0
            ) || 0,
        },
      };

      setHistory((previous) => {
        const day =
          previous[selectedDate] ||
          createEmptyDay();

        return {
          ...previous,

          [selectedDate]: {
            ...day,

            [mealType]: [
              ...(day[mealType] || []),
              meal,
            ],
          },
        };
      });
    } catch (err) {
      console.error(err);

      setError(
        err?.message ||
          "Something went wrong while analysing the meal."
      );
    } finally {
      setUploadingMealType(null);
      activeMealTypeRef.current = null;

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  /* ============================================================
     DELETE
     ============================================================ */

  function deleteMeal(mealType, mealId) {
    setHistory((previous) => {
      const day =
        previous[selectedDate] ||
        createEmptyDay();

      return {
        ...previous,

        [selectedDate]: {
          ...day,

          [mealType]: (day[mealType] || []).filter(
            (meal) => meal.id !== mealId
          ),
        },
      };
    });
  }

  /* ============================================================
     DATE
     ============================================================ */

  function changeDay(amount) {
    const [year, month, day] =
      selectedDate.split("-").map(Number);

    const date = new Date(
      year,
      month - 1,
      day
    );

    date.setDate(date.getDate() + amount);

    const newDate = [
      date.getFullYear(),
      String(date.getMonth() + 1).padStart(2, "0"),
      String(date.getDate()).padStart(2, "0"),
    ].join("-");

    setSelectedDate(newDate);
    setError("");
  }

  function goToToday() {
    setSelectedDate(today);
    setError("");
  }

  return (
    <div className="dashboard">

      {/* Hidden image picker */}

      <input
        ref={fileInputRef}
        className="hidden-file-input"
        type="file"
        accept="image/*"
        onChange={handlePictureSelected}
      />

      {/* ================= HEADER ================= */}

      <header className="dashboard-header">
        <div className="brand">
          <div className="brand-mark">N</div>
          <span>nutriwise</span>
        </div>

        <div className="date-navigation">
          <button
            type="button"
            onClick={() => changeDay(-1)}
          >
            ←
          </button>

          <input
            className="date-input"
            type="date"
            value={selectedDate}
            onChange={(e) =>
              setSelectedDate(e.target.value)
            }
          />

          <button
            type="button"
            onClick={() => changeDay(1)}
          >
            →
          </button>
        </div>
      </header>

      {/* ================= CONTENT ================= */}

      <main className="dashboard-content">
        <div className="dashboard-grid">

          {/* ================= LEFT ================= */}

          <section className="main-column">

            <div className="date-label">
              {formatDateLong(
                selectedDate
              ).toUpperCase()}
            </div>

            <h1>Meal summary.</h1>

            <p className="summary-description">
              {totalFoodWeight > 0
                ? `About ${totalFoodWeight} g of food logged.`
                : "No meals have been logged for this date yet."}
            </p>

            {error && (
              <div className="error-message">
                {error}
              </div>
            )}

            {/* ================= NUTRITION ================= */}

            <section className="nutrition-summary">

              <div className="calorie-card">
                <span className="metric-label">
                  CALORIES
                </span>

                <strong>
                  {totals.calories}
                </strong>

                <span className="metric-unit">
                  kcal
                </span>
              </div>

              <div className="macro-grid">

                <div className="macro-card">
                  <span className="metric-label">
                    PROTEIN
                  </span>

                  <strong>
                    {totals.protein}
                  </strong>

                  <span className="metric-unit">
                    g
                  </span>
                </div>

                <div className="macro-card">
                  <span className="metric-label">
                    CARBS
                  </span>

                  <strong>
                    {totals.carbs}
                  </strong>

                  <span className="metric-unit">
                    g
                  </span>
                </div>

                <div className="macro-card">
                  <span className="metric-label">
                    FAT
                  </span>

                  <strong>
                    {totals.fat}
                  </strong>

                  <span className="metric-unit">
                    g
                  </span>
                </div>

                <div className="macro-card">
                  <span className="metric-label">
                    FIBER
                  </span>

                  <strong>
                    {totals.fiber}
                  </strong>

                  <span className="metric-unit">
                    g
                  </span>
                </div>

              </div>
            </section>

            {/* ================= MEALS ================= */}

            <section className="meal-sections">

              {MEAL_TYPES.map(
                ({ key, label, number }) => {
                  const meals =
                    currentDay[key] || [];

                  const uploading =
                    uploadingMealType === key;

                  return (
                    <section
                      className="meal-section"
                      key={key}
                    >

                      <div className="meal-section-header">

                        <div className="meal-heading">
                          <span className="section-number">
                            {number}
                          </span>

                          <h2>{label}</h2>
                        </div>

                        <button
                          className="picture-button"
                          type="button"
                          disabled={
                            uploadingMealType !== null
                          }
                          onClick={() =>
                            openPicturePicker(key)
                          }
                        >
                          {uploading
                            ? "Analysing..."
                            : "+ Picture"}
                        </button>

                      </div>

                      {meals.length === 0 ? (

                        <EmptyMealBox
                          onPicture={() =>
                            openPicturePicker(key)
                          }
                        />

                      ) : (

                        <div className="logged-meals">

                          {meals.map((meal) => (
                            <MealCard
                              key={meal.id}
                              meal={meal}
                              onDelete={(id) =>
                                deleteMeal(
                                  key,
                                  id
                                )
                              }
                            />
                          ))}

                          <button
                            type="button"
                            className="add-another-picture"
                            onClick={() =>
                              openPicturePicker(key)
                            }
                          >
                            + Add another picture
                          </button>

                        </div>

                      )}

                    </section>
                  );
                }
              )}

            </section>

            <p className="disclaimer">
              Portion sizes and nutrition information
              are estimates based on the available
              food information.
            </p>

          </section>

          {/* ================= HISTORY ================= */}

          <aside className="history-card">

            <div className="history-top">

              <div>
                <span className="history-label">
                  HISTORY
                </span>

                <h3>Previous days</h3>
              </div>

              <button
                className="today-button"
                type="button"
                onClick={goToToday}
              >
                Today
              </button>

            </div>

            <label className="history-date-label">
              CHECK ANOTHER DATE
            </label>

            <input
              className="history-date-input"
              type="date"
              value={selectedDate}
              onChange={(e) =>
                setSelectedDate(e.target.value)
              }
            />

            {/* DATE HISTORY */}

            <div className="history-list">

              {historyDates.length === 0 ? (

                <div className="history-empty">

                  <div className="history-plus">
                    +
                  </div>

                  <span>
                    Previous meals will appear
                    here.
                  </span>

                </div>

              ) : (

                historyDates
                  .slice(0, 7)
                  .map((date) => {

                    const day =
                      history[date];

                    const count =
                      getMealCount(day);

                    const isSelected =
                      date === selectedDate;

                    return (
                      <button
                        type="button"
                        key={date}
                        className={`history-item ${
                          isSelected
                            ? "selected"
                            : ""
                        }`}
                        onClick={() =>
                          setSelectedDate(date)
                        }
                      >

                        <span>
                          {formatDateShort(
                            date
                          )}
                        </span>

                        <strong>
                          {count}{" "}
                          {count === 1
                            ? "meal"
                            : "meals"}
                        </strong>

                      </button>
                    );
                  })

              )}

            </div>

          </aside>

        </div>
      </main>

    </div>
  );
}