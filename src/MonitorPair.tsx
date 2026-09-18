import { publicAsset } from './publicAsset';
import { useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import './monitor-pair.css';

export function useCanvasScale(width: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const resize = () => setScale(element.clientWidth / width);
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(element);
    return () => observer.disconnect();
  }, [width]);
  return { ref, scale };
}

export default function MonitorPair({ left, right }: { left: ReactNode; right: ReactNode }) {
  const { ref, scale } = useCanvasScale(1300);
  return <div ref={ref} className="mc-monitor-pair" aria-hidden="true" inert>
    <div className="mc-monitor-canvas" style={{ transform: `scale(${scale})` }}>
      <div className="mc-device mc-device-back">
        <img className="mc-hardware" src={publicAsset('/assets/imgDisplay01.webp')} alt="" width="4096" height="2816" loading="lazy" />
        <div className="mc-screen mc-screen-back">{left}</div>
      </div>
      <div className="mc-device mc-device-front">
        <img className="mc-hardware" src={publicAsset('/assets/imgDisplay02.webp')} alt="" width="4096" height="2816" loading="lazy" />
        <div className="mc-screen mc-screen-front">{right}</div>
      </div>
    </div>
  </div>;
}
