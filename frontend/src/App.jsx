import { useCallback, useEffect, useRef, useState } from 'react';

import { getEvaluation, getHealth, getVariants, postLandmarks, postPredict } from './api.js';
import Navbar from './components/Navbar.jsx';
import Hero from './components/Hero.jsx';
import About from './components/About.jsx';
import Predict from './components/Predict.jsx';
import Landmarks from './components/Landmarks.jsx';
import Compare from './components/Compare.jsx';
import Evaluation from './components/Evaluation.jsx';
import Team from './components/Team.jsx';
import Footer from './components/Footer.jsx';
import Toasts from './components/Toasts.jsx';

const SECTIONS = [
  { id: 'home', label: 'Home' },
  { id: 'about', label: 'About' },
  { id: 'predict', label: 'Predict' },
  { id: 'landmarks', label: 'Landmarks' },
  { id: 'compare', label: 'Compare' },
  { id: 'evaluate', label: 'Evaluation' },
  { id: 'team', label: 'Team' },
];

const FALLBACK_VARIANTS = [
  { id: 'semantic_left_eye', label: 'Left Eye' },
  { id: 'semantic_right_eye', label: 'Right Eye' },
  { id: 'semantic_left_eyebrow', label: 'Left Eyebrow' },
  { id: 'semantic_right_eyebrow', label: 'Right Eyebrow' },
  { id: 'semantic_nose', label: 'Nose' },
  { id: 'semantic_mouth', label: 'Mouth' },
  { id: 'irregular_shape', label: 'Irregular Blob (center)' },
  { id: 'irregular_shape_left', label: 'Irregular Blob (left)' },
  { id: 'irregular_shape_right', label: 'Irregular Blob (right)' },
];

const FALLBACK_BLENDS = [
  { id: 'feather', label: 'Fast alpha feathering' },
  { id: 'seamless', label: 'Seamless Poisson clone' },
];

export default function App() {
  // --- UI state ---
  const [dark, setDark] = useState(() => localStorage.getItem('morphai-theme') === 'dark');
  const [active, setActive] = useState('home');
  const [showTop, setShowTop] = useState(false);
  const [toasts, setToasts] = useState([]);

  // --- backend state ---
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);
  const [variants, setVariants] = useState(FALLBACK_VARIANTS);
  const [blends, setBlends] = useState(FALLBACK_BLENDS);
  const [evaluation, setEvaluation] = useState(null);
  const [evaluationError, setEvaluationError] = useState(null);

  // --- workflow state ---
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [options, setOptions] = useState({ variant: 'semantic_nose', blend: 'feather', computeIdentity: true });
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState(null);
  const [landmarkResult, setLandmarkResult] = useState(null);
  const [landmarkBusy, setLandmarkBusy] = useState(false);
  const [landmarkError, setLandmarkError] = useState(null);

  const toastId = useRef(0);

  const notify = useCallback((message, type = 'info') => {
    const id = ++toastId.current;
    setToasts((current) => [...current, { id, message, type }]);
    setTimeout(() => setToasts((current) => current.filter((t) => t.id !== id)), 4500);
  }, []);

  const dismissToast = useCallback((id) => {
    setToasts((current) => current.filter((t) => t.id !== id));
  }, []);

  // --- theme ---
  useEffect(() => {
    document.body.classList.toggle('dark-mode', dark);
    localStorage.setItem('morphai-theme', dark ? 'dark' : 'light');
  }, [dark]);

  // --- initial backend handshake ---
  useEffect(() => {
    getHealth()
      .then((data) => {
        setHealth(data);
        setHealthError(null);
        if (!data.model_loaded) {
          notify(data.load_error || 'Backend reachable but the model is not loaded.', 'warning');
        }
        if (!data.arcface_available) {
          setOptions((current) => ({ ...current, computeIdentity: false }));
        }
      })
      .catch((err) => {
        setHealthError(err.message);
        notify(`Cannot reach the backend: ${err.message}`, 'error');
      });

    getVariants()
      .then((data) => {
        if (data.variants?.length) setVariants(data.variants);
        if (data.blends?.length) setBlends(data.blends);
        if (data.default) setOptions((current) => ({ ...current, variant: data.default }));
      })
      .catch(() => {
        /* fallback list already in state */
      });

    getEvaluation()
      .then(setEvaluation)
      .catch((err) => setEvaluationError(err.message));
  }, [notify]);

  // --- scroll spy + back-to-top ---
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible) setActive(visible.target.id);
      },
      { rootMargin: '-40% 0px -50% 0px', threshold: [0, 0.25, 0.5, 1] }
    );

    SECTIONS.forEach(({ id }) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    const onScroll = () => setShowTop(window.scrollY > 400);
    window.addEventListener('scroll', onScroll, { passive: true });

    return () => {
      observer.disconnect();
      window.removeEventListener('scroll', onScroll);
    };
  }, []);

  // Revokes the previous object URL whenever it is replaced, and on unmount.
  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl]);

  // --- landmark detection ---
  const detectLandmarks = useCallback(
    async (target) => {
      const source = target ?? file;
      if (!source) return;

      setLandmarkBusy(true);
      setLandmarkError(null);
      try {
        const data = await postLandmarks(source);
        setLandmarkResult(data);
        if (!data.face_detected) notify('No face detected by MediaPipe in this image.', 'warning');
      } catch (err) {
        setLandmarkError(err.message);
        setLandmarkResult(null);
      } finally {
        setLandmarkBusy(false);
      }
    },
    [file, notify]
  );

  // --- upload ---
  const handleFile = useCallback(
    (selected) => {
      if (!selected.type.startsWith('image/')) {
        notify('Please upload an image file.', 'error');
        return;
      }

      setPreviewUrl(URL.createObjectURL(selected));
      setFile(selected);
      setPrediction(null);
      setPredictError(null);
      setLandmarkResult(null);
      setLandmarkError(null);
      notify('Image uploaded successfully!', 'success');

      // Landmarks are cheap, so the visualisation section fills in right away.
      detectLandmarks(selected);
    },
    [detectLandmarks, notify]
  );

  const handleReset = useCallback(() => {
    setPreviewUrl(null);
    setFile(null);
    setPrediction(null);
    setPredictError(null);
    setLandmarkResult(null);
    setLandmarkError(null);
    notify('Reset successful', 'info');
  }, [notify]);

  // --- prediction ---
  const handlePredict = useCallback(async () => {
    if (!file) {
      notify('Please upload an image first.', 'error');
      return;
    }

    setPredicting(true);
    setPredictError(null);
    try {
      const data = await postPredict(file, options);
      setPrediction(data);
      notify(`Prediction completed in ${data.processing_time}s`, 'success');
    } catch (err) {
      setPredictError(err.message);
      setPrediction(null);
      notify(`Error: ${err.message}`, 'error');
    } finally {
      setPredicting(false);
    }
  }, [file, notify, options]);

  return (
    <>
      <Navbar
        sections={SECTIONS}
        active={active}
        dark={dark}
        onToggleDark={() => setDark((v) => !v)}
        health={health}
        healthError={healthError}
      />

      <main className="container py-4">
        <Hero />
        <About health={health} />
        <Predict
          file={file}
          previewUrl={previewUrl}
          onFile={handleFile}
          onReset={handleReset}
          onPredict={handlePredict}
          busy={predicting}
          prediction={prediction}
          error={predictError}
          options={options}
          setOptions={setOptions}
          variantList={variants}
          blendList={blends}
          identityAvailable={health?.arcface_available ?? false}
        />
        <Landmarks
          file={file}
          result={landmarkResult}
          busy={landmarkBusy}
          error={landmarkError}
          onDetect={() => detectLandmarks()}
        />
        <Compare prediction={prediction} notify={notify} />
        <Evaluation evaluation={evaluation} error={evaluationError} />
        <Team />
        <Footer health={health} />
      </main>

      <Toasts toasts={toasts} onDismiss={dismissToast} />

      {showTop ? (
        <button
          type="button"
          className="btn btn-primary rounded-circle back-to-top"
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          aria-label="Back to top"
        >
          <i className="fas fa-arrow-up" />
        </button>
      ) : null}
    </>
  );
}
