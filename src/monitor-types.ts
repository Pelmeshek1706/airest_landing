export type MonitorVariant = 1 | 2 | 3 | 4;

export interface MonitorScreenProps {
  variant: MonitorVariant;
  side: 'left' | 'right';
  phase: number;
  playing: boolean;
}
