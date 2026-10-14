import { createFileRoute } from "@tanstack/react-router";
import { CardiacDashboard } from '@/components/CardiacDashboard';
export const Route = createFileRoute("/")({
  ssr: false,
  head: () => ({ meta: [
    { title: 'Cardioscope — Cardiovascular Risk Visualizer' },
    { name: 'description', content: 'Explore illustrative cardiac risk with clinical controls, interactive coronary anatomy, and transparent feature contributions.' },
    { property: 'og:title', content: 'Cardioscope — Cardiovascular Risk Visualizer' },
    { property: 'og:description', content: 'An interactive educational cardiac risk workspace with glowing 3D coronary anatomy.' },
    { property: 'og:type', content: 'website' },
    { name: 'twitter:card', content: 'summary_large_image' },
  ] }),
  component: CardiacDashboard,
});
