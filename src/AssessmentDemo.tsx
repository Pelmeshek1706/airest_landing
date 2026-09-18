import { useEffect, useRef, useState } from 'react';
import AssessmentScreens from './AssessmentScreens';
import MonitorPair from './MonitorPair';
import './assessment-demo.css';

const stages = ['Get ready', 'Speak naturally', 'Response summary'];
const durations = [3500, 5500, 8500];

export default function AssessmentDemo() {
  const container = useRef<HTMLDivElement>(null);
  const [phase, setPhase] = useState(0);
  const [paused, setPaused] = useState(false);
  const [visible, setVisible] = useState(false);
  const [pageVisible, setPageVisible] = useState(!document.hidden);
  const [reducedMotion, setReducedMotion] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const playing = !paused && visible && pageVisible && !reducedMotion;

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onPreference = () => setReducedMotion(preference.matches);
    const onVisibility = () => setPageVisible(!document.hidden);
    const observer = new IntersectionObserver(entries => setVisible(entries[0]?.isIntersecting ?? false), { threshold: .15 });
    if (container.current) observer.observe(container.current);
    preference.addEventListener('change', onPreference);
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      observer.disconnect();
      preference.removeEventListener('change', onPreference);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, []);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setTimeout(() => setPhase(value => (value + 1) % stages.length), durations[phase]);
    return () => window.clearTimeout(timer);
  }, [playing, phase]);

  return <div className="assessment-demo" ref={container}>
    <MonitorPair
      left={<AssessmentScreens variant={1} side="left" phase={phase} playing={playing} />}
      right={<AssessmentScreens variant={1} side="right" phase={phase} playing={playing} />} />
    <div className="assessment-demo-controls">
      <button type="button" className="assessment-demo-play" disabled={reducedMotion}
        onClick={() => setPaused(value => !value)} aria-label={paused ? 'Play assessment demo' : 'Pause assessment demo'}>
        <span aria-hidden="true">{paused || reducedMotion ? '▷' : 'Ⅱ'}</span>
        {reducedMotion ? 'Reduced motion' : paused ? 'Play' : 'Pause'}
      </button>
      <div role="group" aria-label="Assessment demo scene" className="assessment-demo-scenes">
        {stages.map((label, index) => <button type="button" key={label} aria-pressed={phase === index}
          onClick={() => { setPhase(index); setPaused(true); }}><span aria-hidden="true" />{label}</button>)}
      </div>
      <span className="assessment-demo-caption">Illustrative demo</span>
    </div>
    <span className="assessment-demo-description">Example of a guided verbal-fluency task, voice and facial-signal capture, and a response summary with illustrative analytics.</span>
  </div>;
}
