import { publicAsset } from './publicAsset';
import { useRef, useState, type CSSProperties } from 'react';
import ContactForm from './ContactForm';
import SignalAnimation from './SignalAnimation';
import AssessmentDemo from './AssessmentDemo';

const asset = (name: string, extension = 'svg') => publicAsset(`assets/${name}.${extension === 'png' ? 'webp' : extension}`);
const heroWatercolorCrop: [number, number, number, number] = [113.41, 564.7, 0, -13.7];

function ArtCrop({ className, image, crop }: {
  className: string;
  image: string;
  crop: [number, number, number, number];
}) {
  const [width, height, left, top] = crop;
  return <div className={`art-crop ${className}`} aria-hidden="true">
    <img src={asset(image, 'png')} alt="" loading={className === 'hero-paint' ? 'eager' : 'lazy'}
      style={{ width: `${width}%`, height: `${height}%`, left: `${left}%`, top: `${top}%` }} />
  </div>;
}

function BrandDot({ variant = 'imgDot' }: { variant?: string }) {
  return <img className="brand-dot" src={asset(variant)} alt="" aria-hidden="true" />;
}

function PartnerButton({ children = 'Partner with', secondary = false }: {
  children?: string;
  secondary?: boolean;
}) {
  return <a className={`button ${secondary ? 'button-secondary' : 'button-primary'}`}
    href={secondary ? '#how-it-works' : '#contact'}>
    <span>{children}</span>
    {!secondary && <img src={asset('imgGroup10')} width="120" height="32" alt="AIREST" />}
  </a>;
}

function Hero() {
  return <section className="hero screen" aria-labelledby="hero-title">
    <ArtCrop className="hero-paint" image="imgWatercolor1" crop={heroWatercolorCrop} />
    <img className="hero-orbit" src={asset('imgCursor')} width="42" height="42" alt="" aria-hidden="true" />
    <div className="hero-copy">
      <h1 id="hero-title" className="display-heading">
        <span>Turning human behaviour</span>
        <span>into measurable <strong>signals</strong><BrandDot /></span>
      </h1>
      <p>Multimodal AI for mental-health screening and clinical decision support.</p>
    </div>
  </section>;
}

function Overview() {
  return <section className="overview screen" id="assessment-demo" aria-label="Introducing AIREST">
    <div className="overview-paint-tail" aria-hidden="true">
      <ArtCrop className="overview-paint" image="imgWatercolor1" crop={heroWatercolorCrop} />
    </div>
    <div className="monitor-composition">
      <AssessmentDemo />
    </div>
    <div className="overview-copy">
      <img className="wordmark" src={asset('imgLogo')} width="290" height="78" alt="AIREST" loading="lazy" />
      <p>Combines voice, language, facial behaviour, gaze, attention and behavioural responses in a short standardised assessment designed to support screening for PTSD, depressive episode and generalised anxiety disorder.</p>
    </div>
  </section>;
}

function Assessment() {
  return <section className="assessment screen" aria-label="A short assessment on a standard computer">
    <div className="assessment-equation">
      <p>10–15 min<br />assessment</p>
      <img src={asset('imgVectorStroke')} width="76" height="76" alt="plus" loading="lazy" />
      <p>Standard<br />computer</p>
    </div>
    <div className="signals-illustration">
      <img src={asset('imgIllustration')} width="1216" height="507" alt="Two faces formed from complementary clusters of golden signals" loading="lazy" />
      <div className="condition-labels"><span>DEPRESSION</span><span>PTSD</span><span>GAD</span></div>
    </div>
    <div className="assessment-buttons">
      <PartnerButton /><PartnerButton secondary>See how it works</PartnerButton>
    </div>
  </section>;
}

const processCards = [
  { title: 'Standardised assessment', description: <>The person completes a short series of voice, visual and behavioural tasks on a regular laptop or desktop computer.</> },
  { title: 'Multimodal analysis', description: <>The system analyses complementary categories of digital biomarkers.</>, icons: true },
  { title: 'Screening outputs', description: <>Overall screening result<br />possible presence of one or more target disorders plus condition-specific results for:<br /><strong>PTSD. Depressive episode.<br />Generalised anxiety disorder.</strong></> },
  { title: 'Clinical decision support', description: <>A qualified healthcare professional interprets the AIREST result alongside the clinical interview, medical history, validated questionnaires and other available information.</> },
];

function Process() {
  return <section id="how-it-works" className="process" aria-labelledby="process-title">
    <ArtCrop className="process-paint" image="imgWatercolorBg" crop={[336.65, 468.11, -31, -66.86]} />
    <div className="split-layout process-layout">
      <h2 id="process-title" className="section-intro"><span>One assessment.</span><span>Multiple digital signals.</span><span>Structured clinical information.</span></h2>
      <ol className="process-grid">
        {processCards.map((card, index) => <li className="glass-card process-card" key={card.title}>
          <div><span className="step-number" aria-hidden="true">0{index + 1}</span><h3>{card.title}</h3></div>
          {card.icons && <div className="signal-icons" role="img" aria-label="Speech, behaviour, audio, language and gaze">
            {['imgIcon', 'imgIcon1', 'imgIcon2', 'imgIcon3', 'imgIcon4'].map(icon => <img src={asset(icon)} width="98" height="98" key={icon} alt="" loading="lazy" />)}
          </div>}
          <p>{card.description}</p>
        </li>)}
      </ol>
    </div>
  </section>;
}

function Multidimensional() {
  return <section className="dimensions screen" id="multidimensional" aria-labelledby="dimensions-title">
    <div className="dimensions-copy">
      <h2 id="dimensions-title" className="display-heading">
        <span>Mental health is</span>
        <span><strong>multidimensional</strong><BrandDot variant="imgUnion1" /></span>
        <span>Its assessment should be too.</span>
      </h2>
      <div className="dimensions-body">
        <p>Mental state can influence more than self-reported symptoms. Changes may also be reflected in speech, language, facial behaviour, attention and response patterns.</p>
        <p><strong>AIREST</strong> is being developed to combine complementary signals rather than depend on a single questionnaire, biomarker or data stream.</p>
      </div>
    </div>
    <SignalAnimation />
  </section>;
}

const clinicianPoints = [
  { title: 'Screens', content: <p>Standardised screening for possible PTSD, depressive episode and generalised anxiety disorder.</p> },
  { title: 'Supports', content: <p>Provides structured information that can support the decision on whether further psychiatric assessment is required.</p> },
  { title: 'Keeps clinicians in control', content: <><p>AIREST does not independently establish a diagnosis, prescribe treatment or replace clinical judgement.</p><p>Final diagnostic and treatment decisions remain with qualified healthcare professionals.</p></> },
];

function Clinicians() {
  return <section className="clinicians screen" aria-labelledby="clinicians-title">
    <ArtCrop className="clinician-paint" image="imgPicture" crop={[496, 549.06, -305.33, 0]} />
    <h2 className="display-heading" id="clinicians-title"><span>Built to support clinicians —</span><span><strong>not replace them</strong><BrandDot variant="imgUnion" /></span></h2>
    <div className="three-columns clinician-points">
      {clinicianPoints.map(point => <article key={point.title}><h3>{point.title}</h3>{point.content}</article>)}
    </div>
  </section>;
}

const researchCards = [
  { title: <>Ukrainian-language<br />biomarker technology</>, description: 'Language-dependent speech and language processing components have undergone technical validation for Ukrainian.' },
  { title: 'Digital biomarker research', description: 'Our research evaluates acoustic, linguistic and behavioural features associated with mental-health assessment.' },
  { title: 'Scientific publications', description: 'The technology is being developed on the basis of our published and ongoing scientific research in digital biomarkers and machine learning.' },
  { title: 'Multimodal prototype', description: 'AIREST combines multiple signal families within a unified assessment platform.' },
];

function Research() {
  return <section className="research" id="research" aria-labelledby="research-title">
    <div className="research-art" aria-hidden="true">
      <ArtCrop className="research-paint research-paint-one" image="imgScan0064" crop={[396.27, 519.8, -245.54, -75.25]} />
      <ArtCrop className="research-paint research-paint-two" image="imgScan0064" crop={[396.27, 519.8, -245.54, -75.25]} />
      <ArtCrop className="research-paint research-paint-three" image="imgScan0064" crop={[396.27, 519.8, -245.54, -75.25]} />
      <ArtCrop className="research-paint research-paint-four" image="imgScan0064" crop={[396.27, 702.92, -158.31, -117.08]} />
      <ArtCrop className="research-paint research-paint-five" image="imgScan0064" crop={[286.85, 452.76, -15.2, -65.62]} />
    </div>
    <div className="split-layout research-layout">
      <h2 className="section-intro" id="research-title">Built on science.<br />Moving into clinical validation.</h2>
      <div className="research-grid">{researchCards.map((card, i) => <article className="glass-card research-card" key={i}><h3>{card.title}</h3><p>{card.description}</p></article>)}</div>
    </div>
    <div className="clinical-programme">
      <div className="programme-left">
        <div className="programme-intro">
          <img className="wordmark" src={asset('imgGroup12')} width="290" height="78" alt="AIREST" loading="lazy" />
          <p>is now entering prospective clinical development using native Ukrainian clinical data. The programme includes:</p>
        </div>
        {['Native Ukrainian clinical data', 'PTSD · Depressive episode · GAD', 'Regulatory-readiness documentation'].map(text => <p className="glass-card programme-point" key={text}>{text}</p>)}
      </div>
      <div className="programme-right">
        {['Cross-sectional study', 'Multiple clinical sites', 'Reference-standard assessment by qualified psychiatrists', 'Model development and validation', 'Usability and safety evaluation'].map(text => <p className="glass-card programme-point" key={text}>{text}</p>)}
      </div>
    </div>
  </section>;
}

const partnerTypes = [
  { title: 'Mental-health clinics', description: 'Standardised screening integrated into clinical assessment pathways.' },
  { title: 'Telehealth providers', description: 'Structured digital screening as part of remote clinical care.' },
  { title: 'Hospitals & rehabilitation services', description: 'Support assessment and referral workflows where specialist capacity is limited.' },
  { title: 'Research institutions & CROs', description: 'Multimodal digital-biomarker collection, clinical research and validation.' },
];

function Partners() {
  const track = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ start: true, end: false });
  const move = (direction: number) => {
    const element = track.current;
    if (!element) return;
    const card = element.querySelector('article');
    const gap = parseFloat(getComputedStyle(element).gap) || 24;
    element.scrollBy({ left: direction * ((card?.getBoundingClientRect().width ?? 320) + gap),
      behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' });
  };
  return <section className="partners" aria-labelledby="partners-title">
    <div className="partners-heading"><h2 className="display-heading" id="partners-title">We work with</h2>
      <div className="carousel-controls"><button aria-label="Previous partner types" disabled={position.start} onClick={() => move(-1)}>←</button><button aria-label="Next partner types" disabled={position.end} onClick={() => move(1)}>→</button></div>
    </div>
    <div className="partners-track" ref={track} tabIndex={0} role="region" aria-label="Organisation types; scroll to see more"
      onScroll={event => { const el = event.currentTarget; setPosition({ start: el.scrollLeft < 4, end: el.scrollLeft + el.clientWidth >= el.scrollWidth - 4 }); }}>
      {partnerTypes.map(partner => <article className="partner-card" key={partner.title}><h3>{partner.title}</h3><p>{partner.description}</p></article>)}
    </div>
  </section>;
}

function Collaborate() {
  return <section className="collaborate screen" aria-labelledby="collaborate-title">
    <div className="collaborate-heading">
      <h2 className="display-heading" id="collaborate-title">We are building the clinical evidence now.</h2>
      <p><strong>AIREST</strong> is looking for organisations and partners interested in helping shape the next generation of mental-health assessment.</p>
    </div>
    <div className="three-columns collaboration-types">
      <article><h3>Clinical partners</h3><p>Clinical sites interested in research participation, data collection or future implementation studies.</p></article>
      <article><h3>Research partners</h3><p>Institutions working in psychiatry, digital biomarkers, multimodal AI and clinical research.</p></article>
      <article><h3>Investors<br />&amp; strategic partners</h3><p>Partners supporting the validation-to-certification pathway and future European market entry.</p></article>
    </div>
    <PartnerButton>Talk to the</PartnerButton>
  </section>;
}

export default function App() {
  return <div className="site" style={{ '--u': 'min(0.0520833333vw, 1px)' } as CSSProperties}>
    <a className="skip-link" href="#main">Skip to content</a>
    <header className="site-header">
      <img src={asset('imgHeader')} width="1920" height="82" alt="" aria-hidden="true" />
      <a className="home-link" href="#main" aria-label="AIREST home" />
    </header>
    <main id="main">
      <Hero /><Overview /><Assessment /><Process /><Multidimensional /><Clinicians /><Research /><Partners /><Collaborate />
      <section className="contact" id="contact" aria-labelledby="contact-form-title">
        <ArtCrop className="contact-paint" image="imgBg" crop={[100, 313.87, 0, -213.87]} />
        <ContactForm />
      </section>
    </main>
  </div>;
}
