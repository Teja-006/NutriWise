import React, { useEffect, useState } from "react";
import {
  Routes,
  Route,
  useNavigate,
  Link,
} from "react-router-dom";

import Dashboard from "./Dashboard";

const STORAGE_KEY = "nutriwise-profile";
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const initialProfile = {
  gender: "",
  height: "",
  weight: "",
  goal: "",
  foodPreference: "",
  allergies: [],
  medicalConditions: [],
};

function loadProfile() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved ? { ...initialProfile, ...JSON.parse(saved) } : initialProfile;
  } catch {
    return initialProfile;
  }
}

function saveProfile(profile) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(profile));
}

function Logo() {
  return (
    <Link to="/" className="logo">
      <span className="logo-mark">
        <span></span>
        <span></span>
        <span></span>
      </span>
      <span>nutriwise</span>
    </Link>
  );
}

function Header() {
  return (
    <header className="site-header">
      <Logo />

      <nav className="desktop-nav">
        <a href="/#about">About</a>
        <a href="/#how">How it works</a>
        <Link to="/privacy">Privacy</Link>
      </nav>

      <Link to="/login" className="header-button">
        Get started
        <span>↗</span>
      </Link>
    </header>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <div className="footer-top">
        <div>
          <Logo />
          <p className="footer-description">
            A simpler way to understand your everyday nutrition.
          </p>
        </div>

        <div className="footer-links">
          <Link to="/privacy">Privacy Policy</Link>
          <Link to="/terms">Terms & Conditions</Link>
        </div>
      </div>

      <div className="footer-bottom">
        <span>© 2026 NutriWise</span>
        <span>Built for a more informed relationship with food.</span>
      </div>
    </footer>
  );
}

function Home() {
  return (
    <div className="page">
      <Header />

      <main>
        <section className="hero">
          <div className="hero-glow glow-one"></div>
          <div className="hero-glow glow-two"></div>

          <div className="hero-content">
            <div className="eyebrow">
              <span className="eyebrow-dot"></span>
              Nutrition, made personal
            </div>

            <h1>
              Know your food.
              <br />
              <span>Know yourself.</span>
            </h1>

            <p className="hero-text">
              NutriWise brings your personal nutrition information into one
              simple experience — designed around you, not a generic formula.
            </p>

            <div className="hero-actions">
              <Link to="/login" className="primary-button">
                Start your journey
                <span>↗</span>
              </Link>

              <a href="#about" className="text-button">
                Explore NutriWise
                <span>↓</span>
              </a>
            </div>
          </div>

          <div className="hero-visual">
            <div className="orb orb-main">
              <div className="orb-inner">
                <div className="leaf leaf-one">✦</div>
                <div className="leaf leaf-two">✦</div>
                <div className="orb-center">N</div>
              </div>
            </div>

            <div className="floating-card card-one">
              <span className="mini-icon">◒</span>
              <div>
                <small>Your profile</small>
                <strong>Personal</strong>
              </div>
            </div>

            <div className="floating-card card-two">
              <span className="mini-icon green">✦</span>
              <div>
                <small>Built around</small>
                <strong>You</strong>
              </div>
            </div>
          </div>
        </section>

        <section className="statement-section" id="about">
          <div className="section-label">01 / THE IDEA</div>

          <div className="statement-grid">
            <h2>
              Nutrition isn't
              <br />
              <em>one-size-fits-all.</em>
            </h2>

            <div>
              <p>
                What works for one person may not work for another. NutriWise
                starts with understanding the person behind the plate.
              </p>

              <p>
                Your information, preferences and goals come together to
                create a clearer picture of your nutritional profile.
              </p>
            </div>
          </div>
        </section>

        <section className="feature-section" id="how">
          <div className="section-label">02 / THE EXPERIENCE</div>

          <div className="feature-heading">
            <h2>
              A clearer picture.
              <br />
              <span>From the start.</span>
            </h2>
          </div>

          <div className="feature-grid">
            <div className="feature-card feature-large">
              <div className="feature-number">01</div>
              <div className="feature-icon">◎</div>
              <h3>Personal profile</h3>
              <p>
                Build a profile around the information that actually matters
                to you.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-number">02</div>
              <div className="feature-icon">⌁</div>
              <h3>Your preferences</h3>
              <p>
                Tell NutriWise what fits your lifestyle and what doesn't.
              </p>
            </div>

            <div className="feature-card dark-card">
              <div className="feature-number">03</div>
              <div className="feature-icon">✦</div>
              <h3>A more informed you</h3>
              <p>
                Understand your nutritional profile through a simple,
                focused experience.
              </p>
            </div>
          </div>
        </section>

        <section className="cta-section">
          <div className="cta-inner">
            <div className="section-label">03 / BEGIN</div>

            <h2>
              Your nutrition.
              <br />
              <span>Your way.</span>
            </h2>

            <p>
              Start with the basics and create your personal NutriWise
              profile.
            </p>

            <Link to="/login" className="primary-button light-button">
              Get started
              <span>↗</span>
            </Link>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}

function Login() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");

  function submit(e) {
    e.preventDefault();

    if (!email.trim() || !email.includes("@")) return;

    navigate("/onboarding/gender");
  }

  return (
    <OnboardingShell step={0}>
      <div className="login-page">
        <div className="login-copy">
          <div className="eyebrow">
            <span className="eyebrow-dot"></span>
            Welcome to NutriWise
          </div>

          <h1>
            Let's make
            <br />
            <span>this personal.</span>
          </h1>

          <p>
            Enter your email to begin creating your NutriWise profile.
          </p>
        </div>

        <form className="login-form" onSubmit={submit}>
          <label>Email address</label>

          <input
            type="email"
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />

          <button className="primary-button full-button" type="submit">
            Continue
            <span>→</span>
          </button>

          <p className="form-note">
            Your information stays within this frontend experience for now.
          </p>
        </form>
      </div>
    </OnboardingShell>
  );
}

function OnboardingShell({ children, step }) {
  const navigate = useNavigate();

  return (
    <div className="onboarding-page">
      <header className="onboarding-header">
        <Logo />

        <div className="progress-wrapper">
          <div className="progress-label">
            <span>YOUR PROFILE</span>
            {step > 0 && <span>{step} / 6</span>}
          </div>

          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{
                width: `${Math.min((step / 6) * 100, 100)}%`,
              }}
            ></div>
          </div>
        </div>

        <button
          className="close-button"
          onClick={() => navigate("/")}
          aria-label="Close"
        >
          ×
        </button>
      </header>

      {children}
    </div>
  );
}

function StepLayout({
  step,
  title,
  description,
  children,
  onContinue,
  canContinue = true,
}) {
  const navigate = useNavigate();

  return (
    <OnboardingShell step={step}>
      <main className="step-container">
        <button className="back-button" onClick={() => navigate(-1)}>
          ← <span>Back</span>
        </button>

        <div className="step-content">
          <div className="step-heading">
            <div className="step-tag">0{step}</div>

            <h1>{title}</h1>

            {description && <p>{description}</p>}
          </div>

          <div className="step-body">{children}</div>

          <button
            className="primary-button continue-button"
            disabled={!canContinue}
            onClick={onContinue}
          >
            Continue
            <span>→</span>
          </button>
        </div>
      </main>
    </OnboardingShell>
  );
}

function GenderStep() {
  const navigate = useNavigate();
  const [profile, setProfile] = useProfile();

  const options = ["Male", "Female", "Prefer not to say"];

  return (
    <StepLayout
      step={1}
      title="How do you identify?"
      description="Choose the option that best represents you."
      canContinue={Boolean(profile.gender)}
      onContinue={() => navigate("/onboarding/body")}
    >
      <div className="option-grid">
        {options.map((option) => (
          <button
            key={option}
            className={`selection-card ${
              profile.gender === option ? "selected" : ""
            }`}
            onClick={() =>
              setProfile((p) => ({
                ...p,
                gender: option,
              }))
            }
          >
            <span className="selection-icon">
              {option === "Male"
                ? "♂"
                : option === "Female"
                ? "♀"
                : "○"}
            </span>

            <span>{option}</span>

            <span className="check">✓</span>
          </button>
        ))}
      </div>
    </StepLayout>
  );
}

function BodyStep() {
  const navigate = useNavigate();
  const [profile, setProfile] = useProfile();

  const height = Number(profile.height);
  const weight = Number(profile.weight);

  const bmi =
    height > 0 && weight > 0
      ? Number((weight / Math.pow(height / 100, 2)).toFixed(1))
      : null;

  const category = getBmiCategory(bmi);

  const valid =
    height >= 100 &&
    height <= 250 &&
    weight >= 20 &&
    weight <= 300;

  return (
    <StepLayout
      step={2}
      title="Let's get your basics."
      description="We'll use your height and weight to calculate your BMI."
      canContinue={valid}
      onContinue={() => navigate("/onboarding/goal")}
    >
      <div className="measurement-grid">
        <div className="input-group">
          <label>Height</label>

          <div className="input-with-unit">
            <input
              type="number"
              min="100"
              max="250"
              placeholder="170"
              value={profile.height}
              onChange={(e) =>
                setProfile((p) => ({
                  ...p,
                  height: e.target.value,
                }))
              }
            />
            <span>cm</span>
          </div>
        </div>

        <div className="input-group">
          <label>Weight</label>

          <div className="input-with-unit">
            <input
              type="number"
              min="20"
              max="300"
              placeholder="65"
              value={profile.weight}
              onChange={(e) =>
                setProfile((p) => ({
                  ...p,
                  weight: e.target.value,
                }))
              }
            />
            <span>kg</span>
          </div>
        </div>
      </div>

      <div className={`bmi-card ${bmi ? "active" : ""}`}>
        <div>
          <span className="bmi-label">YOUR BMI</span>
          <strong>{bmi ?? "—"}</strong>
        </div>

        <div className="bmi-category">
          <span>Category</span>
          <strong>{category || "Enter your details"}</strong>
        </div>
      </div>

      <p className="bmi-note">
        BMI is calculated from your height and weight using the standard BMI
        formula.
      </p>
    </StepLayout>
  );
}

function GoalStep() {
  const navigate = useNavigate();
  const [profile, setProfile] = useProfile();

  const options = [
    "Maintain my current weight",
    "Lose weight",
    "Gain weight",
    "Build healthier habits",
  ];

  return (
    <StepLayout
      step={3}
      title="What are you working towards?"
      description="Choose the goal that feels most relevant to you."
      canContinue={Boolean(profile.goal)}
      onContinue={() => navigate("/onboarding/preference")}
    >
      <div className="vertical-options">
        {options.map((option) => (
          <button
            className={`wide-selection ${
              profile.goal === option ? "selected" : ""
            }`}
            key={option}
            onClick={() =>
              setProfile((p) => ({
                ...p,
                goal: option,
              }))
            }
          >
            <span>{option}</span>
            <span className="arrow">→</span>
          </button>
        ))}
      </div>
    </StepLayout>
  );
}

function PreferenceStep() {
  const navigate = useNavigate();
  const [profile, setProfile] = useProfile();

  const options = ["Vegetarian", "Non-vegetarian", "Vegan", "Other"];

  return (
    <StepLayout
      step={4}
      title="What's your food preference?"
      description="Select the option that best describes how you eat."
      canContinue={Boolean(profile.foodPreference)}
      onContinue={() => navigate("/onboarding/allergies")}
    >
      <div className="option-grid two-column">
        {options.map((option) => (
          <button
            className={`selection-card ${
              profile.foodPreference === option ? "selected" : ""
            }`}
            key={option}
            onClick={() =>
              setProfile((p) => ({
                ...p,
                foodPreference: option,
              }))
            }
          >
            <span className="selection-icon">
              {option === "Vegetarian"
                ? "◌"
                : option === "Non-vegetarian"
                ? "◉"
                : option === "Vegan"
                ? "✿"
                : "＋"}
            </span>

            <span>{option}</span>

            <span className="check">✓</span>
          </button>
        ))}
      </div>
    </StepLayout>
  );
}

function AllergiesStep() {
  const navigate = useNavigate();
  const [profile, setProfile] = useProfile();

  const options = [
    "Milk / Dairy",
    "Nuts",
    "Peanuts",
    "Gluten",
    "Soy",
    "None",
  ];

  function toggle(option) {
    setProfile((p) => {
      if (option === "None") {
        return {
          ...p,
          allergies: p.allergies.includes("None") ? [] : ["None"],
        };
      }

      const withoutNone = p.allergies.filter((x) => x !== "None");

      return {
        ...p,
        allergies: withoutNone.includes(option)
          ? withoutNone.filter((x) => x !== option)
          : [...withoutNone, option],
      };
    });
  }

  return (
    <StepLayout
      step={5}
      title="Any allergies?"
      description="Select anything you need to avoid, or choose none."
      canContinue={profile.allergies.length > 0}
      onContinue={() => navigate("/onboarding/conditions")}
    >
      <div className="check-grid">
        {options.map((option) => (
          <button
            key={option}
            className={`check-card ${
              profile.allergies.includes(option) ? "selected" : ""
            }`}
            onClick={() => toggle(option)}
          >
            <span className="checkbox">
              {profile.allergies.includes(option) ? "✓" : ""}
            </span>
            <span>{option}</span>
          </button>
        ))}
      </div>
    </StepLayout>
  );
}

function ConditionsStep() {
  const navigate = useNavigate();
  const [profile, setProfile] = useProfile();

  const options = [
    "Diabetes",
    "High blood pressure",
    "High cholesterol",
    "Thyroid condition",
    "None",
  ];

  function toggle(option) {
    setProfile((p) => {
      if (option === "None") {
        return {
          ...p,
          medicalConditions: p.medicalConditions.includes("None")
            ? []
            : ["None"],
        };
      }

      const withoutNone = p.medicalConditions.filter((x) => x !== "None");

      return {
        ...p,
        medicalConditions: withoutNone.includes(option)
          ? withoutNone.filter((x) => x !== option)
          : [...withoutNone, option],
      };
    });
  }

  return (
    <StepLayout
      step={6}
      title="Anything else we should know?"
      description="Select any relevant condition, or choose none."
      canContinue={profile.medicalConditions.length > 0}
      onContinue={() => navigate("/complete")}
    >
      <div className="check-grid">
        {options.map((option) => (
          <button
            key={option}
            className={`check-card ${
              profile.medicalConditions.includes(option) ? "selected" : ""
            }`}
            onClick={() => toggle(option)}
          >
            <span className="checkbox">
              {profile.medicalConditions.includes(option) ? "✓" : ""}
            </span>
            <span>{option}</span>
          </button>
        ))}
      </div>
    </StepLayout>
  );
}

function Complete() {
  const navigate = useNavigate();
  const [profile] = useProfile();

  return (
    <OnboardingShell step={6}>
      <main className="complete-page">
        <div className="complete-orb">
          <div>✓</div>
        </div>

        <div className="eyebrow centered">
          <span className="eyebrow-dot"></span>
          Profile complete
        </div>

        <h1>
          You're all set.
          <br />
          <span>Welcome to NutriWise.</span>
        </h1>

        <p>
          Your basic NutriWise profile has been saved on this device.
        </p>

        <div className="summary-card">
          <div>
            <span>Gender</span>
            <strong>{profile.gender || "—"}</strong>
          </div>

          <div>
            <span>BMI</span>
            <strong>
              {profile.height && profile.weight
                ? (
                    Number(profile.weight) /
                    Math.pow(Number(profile.height) / 100, 2)
                  ).toFixed(1)
                : "—"}
            </strong>
          </div>

          <div>
            <span>Goal</span>
            <strong>{profile.goal || "—"}</strong>
          </div>

          <div>
            <span>Preference</span>
            <strong>{profile.foodPreference || "—"}</strong>
          </div>
        </div>

        <button className="primary-button" onClick={() => navigate("/dashboard")}>
          Open my dashboard
          <span>→</span>
        </button>
      </main>
    </OnboardingShell>
  );
}

function prettyName(name) {
  const s = name.replace(/-/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function Analyze() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  function onPick(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Please choose an image file.");
      return;
    }
    setError("");
    setResult(null);
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${API_URL}/predict`, { method: "POST", body });
      if (!res.ok) {
        const msg = await res.json().catch(() => ({}));
        throw new Error(msg.detail || `Server error (${res.status})`);
      }
      setResult(await res.json());
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

  return (
    <OnboardingShell step={6}>
      <main className="analyze-page">
        <div className="step-heading">
          <div className="step-tag">PLATE</div>
          <h1>Show us your plate.</h1>
          <p>
            Upload a top-down photo of your thali. Best results: good light,
            whole plate in frame.
          </p>
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
          <p className="analyze-note">
            This can take 20 to 60 seconds. Hang tight.
          </p>
        )}

        {result && (
          <section className="result-card">
            {result.items.length === 0 ? (
              <p>No dishes detected. Try a clearer, top-down photo.</p>
            ) : (
              <>
                <div className="result-scroll">
                  <table className="result-table">
                    <thead>
                      <tr>
                        <th>Dish</th>
                        <th>g</th>
                        <th>kcal</th>
                        <th>Protein</th>
                        <th>Carbs</th>
                        <th>Fat</th>
                        <th>Fiber</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.items.map((it) => (
                        <tr key={it.id}>
                          <td>{prettyName(it.name)}</td>
                          <td>{it.weight_g?.toFixed(0) ?? "—"}</td>
                          <td>{it.nutrition?.calories_kcal.toFixed(0) ?? "—"}</td>
                          <td>{it.nutrition?.protein_g.toFixed(1) ?? "—"}</td>
                          <td>{it.nutrition?.carbs_g.toFixed(1) ?? "—"}</td>
                          <td>{it.nutrition?.fat_g.toFixed(1) ?? "—"}</td>
                          <td>{it.nutrition?.fiber_g.toFixed(1) ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                    {result.totals && (
                      <tfoot>
                        <tr>
                          <td>Total</td>
                          <td>{result.total_grams.toFixed(0)}</td>
                          <td>{result.totals.calories_kcal.toFixed(0)}</td>
                          <td>{result.totals.protein_g.toFixed(1)}</td>
                          <td>{result.totals.carbs_g.toFixed(1)}</td>
                          <td>{result.totals.fat_g.toFixed(1)}</td>
                          <td>{result.totals.fiber_g.toFixed(1)}</td>
                        </tr>
                      </tfoot>
                    )}
                  </table>
                </div>
                <p className="analyze-note">
                  Portion sizes and nutrition are estimates, not exact
                  measurements.
                </p>
              </>
            )}
          </section>
        )}
      </main>
    </OnboardingShell>
  );
}

function LegalPage({ type }) {
  const isPrivacy = type === "privacy";

  return (
    <div className="legal-page">
      <Header />

      <main className="legal-content">
        <div className="section-label">
          LEGAL / {isPrivacy ? "PRIVACY" : "TERMS"}
        </div>

        <h1>{isPrivacy ? "Privacy Policy" : "Terms & Conditions"}</h1>

        <p className="legal-intro">
          This page is a frontend placeholder for the final NutriWise legal
          document.
        </p>

        {isPrivacy ? (
          <>
            <LegalSection
              title="1. Introduction"
              text="This Privacy Policy placeholder is intended to be replaced with the final privacy policy before launch."
            />

            <LegalSection
              title="2. Information We Collect"
              text="The final document should specify what information NutriWise collects, how it is used, where it is stored, and how long it is retained."
            />

            <LegalSection
              title="3. How Information Is Used"
              text="The final policy should explain the purposes for which information is processed and the applicable user rights."
            />

            <LegalSection
              title="4. Contact"
              text="Final contact information for privacy-related requests should be added before launch."
            />
          </>
        ) : (
          <>
            <LegalSection
              title="1. Acceptance"
              text="This Terms & Conditions placeholder is intended to be replaced with the final terms before launch."
            />

            <LegalSection
              title="2. Use of the Website"
              text="The final terms should describe acceptable use of NutriWise and the responsibilities of users."
            />

            <LegalSection
              title="3. Disclaimer"
              text="The final document should clearly describe the nature and limitations of the NutriWise service."
            />

            <LegalSection
              title="4. Contact"
              text="Final legal contact information should be added before launch."
            />
          </>
        )}
      </main>

      <Footer />
    </div>
  );
}

function LegalSection({ title, text }) {
  return (
    <section className="legal-section">
      <h2>{title}</h2>
      <p>{text}</p>
    </section>
  );
}

function getBmiCategory(bmi) {
  if (!bmi) return "";

  if (bmi < 18.5) return "Underweight";
  if (bmi < 25) return "Normal weight";
  if (bmi < 30) return "Overweight";
  return "Obesity";
}

function useProfile() {
  const [profile, setProfileState] = useState(loadProfile);

  useEffect(() => {
    saveProfile(profile);
  }, [profile]);

  const setProfile = (updater) => {
    setProfileState((current) =>
      typeof updater === "function" ? updater(current) : updater
    );
  };

  return [profile, setProfile];
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/login" element={<Login />} />
      <Route path="/onboarding/gender" element={<GenderStep />} />
      <Route path="/onboarding/body" element={<BodyStep />} />
      <Route path="/onboarding/goal" element={<GoalStep />} />
      <Route path="/onboarding/preference" element={<PreferenceStep />} />
      <Route path="/onboarding/allergies" element={<AllergiesStep />} />
      <Route path="/onboarding/conditions" element={<ConditionsStep />} />
      <Route path="/complete" element={<Complete />} />
      <Route path="/analyze" element={<Analyze />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/privacy" element={<LegalPage type="privacy" />} />
      <Route path="/terms" element={<LegalPage type="terms" />} />
    </Routes>
  );
}

export default App;