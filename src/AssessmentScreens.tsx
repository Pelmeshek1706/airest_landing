import { publicAsset } from './publicAsset';
import type { CSSProperties } from 'react';
import type { MonitorScreenProps } from './monitor-types';
import faceLandmarks from './demo-face-landmarks.json';
import faceConnections from './demo-face-connections.json';
import './assessment-screens.css';

const waveform = [8, 13, 24, 17, 39, 54, 28, 66, 91, 59, 34, 49, 76, 96, 68, 41, 23, 47, 72, 51, 29, 14, 31, 62, 87, 56, 39, 21, 42, 68, 93, 75, 47, 28, 40, 59, 35, 19, 32, 48, 24, 14, 20, 10];
const faceMeshPath = faceConnections.map(([start, end]) => {
  const a = faceLandmarks[start];
  const b = faceLandmarks[end];
  return `M${a.x * 1448},${a.y * 1086}L${b.x * 1448},${b.y * 1086}`;
}).join('');

function Waveform({ quiet = false }: { quiet?: boolean }) {
  return <div className={`as-waveform${quiet ? ' as-waveform-quiet' : ''}`} aria-hidden="true">
    {waveform.map((height, index) => <span key={index} style={{ height: `${height}%`, '--as-delay': `${index * -0.091}s`, '--as-duration': `${0.55 + (index % 6) * 0.09}s` } as CSSProperties} />)}
  </div>;
}

function Header({ label = 'Assessment', step }: { label?: string; step?: string }) {
  return <header className="as-header">
    <img src={publicAsset('/assets/imgLogo.svg')} className="as-logo" alt="AIREST" />
    <span className="as-header-label">{label}</span>
    <span className="as-demo">Demo session{step && <b>{step}</b>}</span>
  </header>;
}

function Check({ className = '' }: { className?: string }) {
  return <svg className={`as-check ${className}`} viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m5 12 4.5 4.5L19 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}

function Portrait({ guide = false, small = false }: { guide?: boolean; small?: boolean }) {
  return <div className={`as-portrait${small ? ' as-portrait-small' : ''}`}>
    <img src={publicAsset('/assets/demo-participant.png')} alt="Illustrative assessment participant" />
    {guide && <svg className="as-face-guide" viewBox="0 0 1448 1086" preserveAspectRatio="xMidYMid slice" fill="none" aria-hidden="true">
      <path d={faceMeshPath} stroke="white" strokeWidth="1.35" opacity=".32" />
      <g fill="white" opacity=".68">{faceLandmarks.map(({ x, y }, i) => <circle key={i} cx={x * 1448} cy={y * 1086} r="1.8" />)}</g>
    </svg>}
    {!small && <span className="as-camera-tag"><span />Camera connected</span>}
  </div>;
}

function AssessmentLeft({ phase }: { phase: number }) {
  return <>
    <Header step="02 / 06" />
    <div className="as-assessment-body">
      <section className="as-task-copy">
        <span className="as-eyebrow">Verbal fluency</span>
        <h2>{phase === 0 ? 'Ready when\nyou are.' : phase === 2 ? 'Nicely done.' : 'Name as many\nanimals as you can.'}</h2>
        <p>{phase === 0 ? 'Speak naturally. We’ll guide you through each short task.' : phase === 2 ? 'Your response has been recorded. Take a moment before the next task.' : 'Say each name out loud. Keep going until the timer finishes.'}</p>
        <div className={`as-recording-pill${phase === 2 ? ' as-done' : ''}`}>
          {phase === 2 ? <Check /> : <span className="as-record-dot" />}
          {phase === 0 ? 'Microphone ready' : phase === 2 ? 'Task complete' : 'Recording your response'}
        </div>
      </section>
      <Portrait />
    </div>
    <footer className="as-recording-footer">
      <div className="as-recording-time"><span>{phase === 0 ? '00:00' : phase === 2 ? '01:00' : '00:24'}</span><small>of 01:00</small></div>
      <Waveform quiet={phase !== 1} />
      <span className={`as-soft-button${phase === 2 ? ' as-gold-button' : ''}`}>{phase === 2 ? 'Continue →' : 'Pause'}</span>
    </footer>
  </>;
}

function AssessmentRight({ phase }: { phase: number }) {
  if (phase === 2) return <AssessmentSummary />;

  return <>
    <Header label="Signal capture" />
    <div className="as-signals-heading"><h2>One response.<br /><em>Multiple signals.</em></h2><span className="as-live-label"><i />{phase === 2 ? 'Captured' : phase === 0 ? 'Ready' : 'Live capture'}</span></div>
    <div className="as-signals-layout">
      <div className="as-face-panel"><Portrait guide /><div className="as-panel-caption"><strong>Facial landmarks</strong><span><Check />{phase === 2 ? 'Captured' : 'Tracking ready'}</span></div></div>
      <div className="as-voice-panel"><div className="as-panel-top"><span className="as-eyebrow">Voice recording</span><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect x="9" y="3" width="6" height="12" rx="3" stroke="currentColor" strokeWidth="1.5"/><path d="M6 11a6 6 0 0 0 12 0M12 17v4m-4 0h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg></div><Waveform quiet={phase !== 1} /><div className="as-voice-bottom"><strong>{phase === 2 ? 'Response captured' : phase === 0 ? 'Listening is ready' : 'Listening to your response'}</strong><span><Check />Microphone connected</span></div></div>
    </div>
    <div className="as-screen-note">Illustrative signal view <span>Verbal fluency · Task 02</span></div>
  </>;
}

const responseTimeline = [
  { seconds: 8, speaking: true }, { seconds: 2, speaking: false },
  { seconds: 11, speaking: true }, { seconds: 3, speaking: false },
  { seconds: 12, speaking: true }, { seconds: 4, speaking: false },
  { seconds: 9, speaking: true }, { seconds: 5, speaking: false },
  { seconds: 6, speaking: true },
];

function AssessmentSummary() {
  return <>
    <Header label="Response summary" />
    <div className="as-summary-content">
      <div className="as-summary-heading">
        <h2>One response.<br /><em>A clearer picture.</em></h2>
        <span className="as-summary-complete"><Check />Task complete</span>
      </div>
      <div className="as-summary-metrics">
        <div className="as-summary-metric as-summary-primary"><strong>18</strong><span>Unique animals</span></div>
        <div className="as-summary-metric"><strong>46<small>s</small></strong><span>Speaking time</span></div>
        <div className="as-summary-metric"><strong>14<small>s</small></strong><span>Time in pauses</span></div>
      </div>
      <div className="as-summary-timing">
        <div className="as-summary-timing-heading"><strong>Response rhythm</strong><span><i />Speech <i className="as-pause-key" />Pauses</span></div>
        <div className="as-summary-track" aria-label="Illustrative 60-second response: 46 seconds of speech and 14 seconds of pauses">
          {responseTimeline.map((segment, index) => <span key={index} className={segment.speaking ? 'as-speech-segment' : 'as-pause-segment'} style={{ flex: segment.seconds }} />)}
        </div>
        <div className="as-summary-axis"><span>0s</span><span>15s</span><span>30s</span><span>45s</span><span>60s</span></div>
      </div>
      <div className="as-summary-capture"><span className="as-summary-capture-icon"><Check /></span><div><strong>Facial landmarks captured</strong><span>Alongside the voice response</span></div><span className="as-summary-task">Verbal fluency · Task 02</span></div>
    </div>
    <div className="as-summary-note">Illustrative demo values<span>Prepared for review</span></div>
  </>;
}

const journeyTasks = [
  { type: 'Reading aloud', prompt: '“The morning light\nfilled the room.”', instruction: 'Read the short passage at your natural pace.', duration: 'Reading task', short: 'A short passage, in your own voice.' },
  { type: 'Name animals', prompt: 'How many animals\ncan you name?', instruction: 'Say as many different animal names as you can.', duration: '60 seconds', short: 'Name as many animals as you can.' },
  { type: 'Words beginning with K', prompt: 'One letter.\nAs many words as you can.', instruction: 'Name words beginning with the letter K.', duration: '60 seconds', short: 'Find words that begin with K.' },
];

function JourneyLeft({ phase }: { phase: number }) {
  const task = journeyTasks[phase];
  return <>
    <Header label="Guided assessment" step={`${String(phase + 1).padStart(2,'0')} / 03`} />
    <div className="as-journey-task">
      <div className="as-journey-task-label"><span className="as-task-number">{String(phase + 1).padStart(2,'0')}</span><div><span className="as-eyebrow">Your next task</span><h3>{task.type}</h3></div><span className="as-task-duration">{task.duration}</span></div>
      <h2 className={phase === 0 ? 'as-reading-prompt' : ''}>{task.prompt}</h2>
      <p>{task.instruction}</p>
      <div className="as-journey-response"><span className="as-record-dot" /><strong>Speak naturally</strong><Waveform /><span className="as-response-time">00:{phase === 0 ? '12' : phase === 1 ? '24' : '08'}</span></div>
      <Portrait small />
    </div>
    <div className="as-journey-footer"><span className="as-journey-footnote">One task at a time. At your pace.</span><div className="as-step-dots">{journeyTasks.map((_, i) => <span key={i} className={i === phase ? 'as-active-dot' : i < phase ? 'as-completed-dot' : ''} />)}</div></div>
  </>;
}

function JourneyRight({ phase }: { phase: number }) {
  return <>
    <Header label="Session progress" />
    <div className="as-protocol-heading"><h2>A little structure.<br /><em>A fuller picture.</em></h2><span className="as-progress-count">0{phase + 1}<small> / 03</small></span></div>
    <div className="as-timeline">
      {journeyTasks.map((task, i) => <div className={`as-timeline-row${i === phase ? ' as-current' : i < phase ? ' as-complete' : ''}`} key={task.type}>
        <div className="as-timeline-marker">{i < phase ? <Check /> : String(i + 1).padStart(2,'0')}</div>
        <div className="as-timeline-copy"><h3>{task.type}</h3><p>{task.short}</p></div>
        <span className="as-timeline-status">{i < phase ? 'Complete' : i === phase ? 'In progress' : 'Up next'}</span>
      </div>)}
    </div>
    <div className="as-protocol-footer"><span><i />Session in progress</span><Waveform /><small>Demo session</small></div>
  </>;
}

export default function AssessmentScreens({ variant, side, phase, playing }: MonitorScreenProps) {
  const scene = Math.max(0, Math.min(2, phase));
  return <div className={`as-screen as-variant-${variant} as-side-${side}${playing ? ' as-playing' : ''} as-phase-${scene}`}>
    {variant === 3 ? side === 'left' ? <JourneyLeft phase={scene} /> : <JourneyRight phase={scene} /> : side === 'left' ? <AssessmentLeft phase={scene} /> : <AssessmentRight phase={scene} />}
  </div>;
}
