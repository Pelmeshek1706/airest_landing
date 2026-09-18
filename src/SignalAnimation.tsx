import { publicAsset } from './publicAsset';
import { useEffect, useRef, useState } from 'react';

const motionPreference = '(prefers-reduced-motion: reduce)';

export default function SignalAnimation() {
  const container = useRef<HTMLDivElement>(null);
  const media = useRef<HTMLVideoElement>(null);
  const togglePlayback = useRef<(() => void) | null>(null);
  const [reducedMotion, setReducedMotion] = useState(() =>
    typeof window !== 'undefined' && window.matchMedia(motionPreference).matches);
  const [hasPlayed, setHasPlayed] = useState(false);
  const [showVideo, setShowVideo] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    const element = container.current;
    const video = media.current;
    if (!element || !video) return;

    const preference = window.matchMedia(motionPreference);
    let motionReduced = preference.matches;
    let visible = false;
    let userPaused = false;
    let failed = false;
    let disposed = false;
    let request = 0;

    setReducedMotion(motionReduced);
    setHasPlayed(false);
    setShowVideo(false);
    setIsPlaying(false);

    const shouldPlay = () => !disposed && visible && !document.hidden
      && !motionReduced && !userPaused && !failed;

    const pause = () => {
      request += 1;
      video.pause();
      if (!disposed) setIsPlaying(false);
    };

    const useFallback = () => {
      if (disposed) return;
      failed = true;
      pause();
      setHasPlayed(false);
      setShowVideo(false);
    };

    const reconcilePlayback = () => {
      if (!shouldPlay()) {
        pause();
        return;
      }

      // Assign the source only when motion is permitted and the artwork is visible.
      if (!video.hasAttribute('src')) video.src = publicAsset('/assets/face-morph.mp4');
      const attempt = ++request;
      void video.play().catch(() => {
        // Pausing, changing preferences, or unmounting can interrupt a pending play.
        if (!disposed && attempt === request && shouldPlay()) useFallback();
      });
    };

    const onPlaying = () => {
      if (!shouldPlay()) {
        pause();
        return;
      }
      setHasPlayed(true);
      setShowVideo(true);
      setIsPlaying(true);
    };

    const onPause = () => {
      if (!disposed) setIsPlaying(false);
    };

    const onPreferenceChange = () => {
      motionReduced = preference.matches;
      setReducedMotion(motionReduced);
      if (motionReduced) {
        pause();
        setShowVideo(false);
        video.removeAttribute('src');
        video.load();
      } else {
        reconcilePlayback();
      }
    };

    togglePlayback.current = () => {
      userPaused = !video.paused;
      reconcilePlayback();
    };

    video.addEventListener('playing', onPlaying);
    video.addEventListener('pause', onPause);
    video.addEventListener('error', useFallback);
    preference.addEventListener('change', onPreferenceChange);
    document.addEventListener('visibilitychange', reconcilePlayback);

    const observer = new IntersectionObserver(entries => {
      const entry = entries[entries.length - 1];
      if (!entry) return;
      visible = entry.isIntersecting;
      reconcilePlayback();
    });
    observer.observe(element);

    return () => {
      disposed = true;
      request += 1;
      togglePlayback.current = null;
      observer.disconnect();
      preference.removeEventListener('change', onPreferenceChange);
      document.removeEventListener('visibilitychange', reconcilePlayback);
      video.removeEventListener('playing', onPlaying);
      video.removeEventListener('pause', onPause);
      video.removeEventListener('error', useFallback);
      video.pause();
      video.removeAttribute('src');
      video.load();
    };
  }, []);

  return <div className="mouth-art" ref={container} data-animated={showVideo && !reducedMotion}>
    <div className="mouth-art-poster">
      <img src={publicAsset('/assets/imgImage158.webp')} width="491" height="589"
        alt="Human speech pictured through the AIREST symbol" loading="lazy" />
    </div>
    <video className="mouth-art-video" ref={media} width="400" height="400"
      muted loop playsInline preload="none" aria-hidden="true" tabIndex={-1} />
    {hasPlayed && !reducedMotion && <button className="mouth-art-toggle" type="button"
      onClick={() => togglePlayback.current?.()}
      aria-label={isPlaying ? 'Pause animation' : 'Play animation'}>
      <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false">
        {isPlaying
          ? <><rect x="6" y="4" width="4" height="16" rx="1" /><rect x="14" y="4" width="4" height="16" rx="1" /></>
          : <path d="M7 4.5a1 1 0 0 1 1.5-.86l12 7.5a1 1 0 0 1 0 1.72l-12 7.5A1 1 0 0 1 7 19.5z" />}
      </svg>
    </button>}
  </div>;
}
