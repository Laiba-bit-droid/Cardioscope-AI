import { Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Environment, Lightformer, OrbitControls } from '@react-three/drei';
import { Bloom, EffectComposer } from '@react-three/postprocessing';
import * as THREE from 'three';
import type { OrbitControls as OrbitControlsType } from 'three-stdlib';
import { Button } from '@/components/ui/button';
import { Layers, Maximize, Minus, Plus, RotateCcw, Rotate3D } from 'lucide-react';
import { riskLevel, type Vessel } from '@/lib/cardiac-risk';

const colorTokens = ['heart-body','heart-emission','heart-highlight','heart-artery','pink','risk-low','risk-moderate','risk-high','foreground'] as const;
type Palette = Record<(typeof colorTokens)[number], string>;
type Point = [number, number, number];
const paths: Record<Vessel, Point[]> = {
 LAD: [[.18,.98,.49],[.05,.65,.81],[.13,.3,.97],[.3,-.08,.94],[.42,-.48,.76],[.43,-.95,.38],[.37,-1.2,.13]],
 LCX: [[.18,.98,.49],[-.22,.91,.7],[-.66,.76,.69],[-.96,.48,.42],[-1.06,.12,.01],[-.89,-.19,-.48],[-.51,-.3,-.71]],
 RCA: [[-.37,1,.41],[-.64,.71,.64],[-.84,.4,.64],[-.8,.09,.74],[-.52,-.21,.87],[-.17,-.39,.92],[.09,-.63,.8]],
};
function Tube({ points, radius, color, glow = 0, onClick }: { points: Point[]; radius: number; color: string; glow?: number; onClick?: () => void }) {
 const curve = useMemo(() => new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(...p))), [points]);
 return <mesh onClick={e => {if (onClick) { e.stopPropagation(); onClick(); }}}><tubeGeometry args={[curve, 56, radius, 10, false]} /><meshStandardMaterial color={color} emissive={color} emissiveIntensity={glow} roughness={.3} metalness={.22} /></mesh>;
}
function tissueGeometry() {
 const geo = new THREE.SphereGeometry(1, 80, 64);
 const p = geo.getAttribute('position');
 for(let i=0;i<p.count;i++) {
  const x=p.getX(i), y=p.getY(i), z=p.getZ(i);
  const taper = .69 + .3 * (y + 1) / 2;
  const groove = .05 * Math.exp(-Math.pow((x-.18)/.1,2)) * Math.max(z,0);
  const ripple = .012 * Math.sin(x*28+y*12) * Math.sin(z*25-y*10);
  p.setXYZ(i, x*taper + .3*(1-y)/2, y*1.23-.18, z*(.74+.17*(y+1)/2) - groove + ripple);
 }
 geo.computeVertexNormals(); return geo;
}
function Heart({ colors, risks, onSelect, reduced }: { colors: Palette; risks: Record<Vessel, number>; onSelect: (v: Vessel) => void; reduced: boolean }) {
 const ref=useRef<THREE.Group>(null); const time=useRef(0);
 const geometry=useMemo(tissueGeometry,[]);
 const bump=useMemo(() => { const size=128; const data=new Uint8Array(size*size); for(let y=0;y<size;y++) for(let x=0;x<size;x++) data[y*size+x]=Math.round(128+30*Math.sin(x*.7+y*.2)*Math.sin(y*.55)+12*Math.sin(x*1.7+y*2.1)); const texture=new THREE.DataTexture(data,size,size,THREE.RedFormat); texture.wrapS=texture.wrapT=THREE.RepeatWrapping; texture.needsUpdate=true; return texture; },[]);
 useEffect(() => () => { geometry.dispose(); bump.dispose(); },[geometry,bump]);
 useFrame((_,delta) => { if(!ref.current || reduced) return; time.current+=Math.min(delta,.05); const pulse=1+.013*Math.pow(Math.max(0,Math.sin(time.current*7.5)),8); ref.current.scale.setScalar(pulse); });
 const material = <meshPhysicalMaterial color={colors['heart-body']} emissive={colors['heart-emission']} emissiveIntensity={.32} roughness={.34} metalness={.18} clearcoat={.55} clearcoatRoughness={.3} bumpMap={bump} bumpScale={.026} />;
 return <group ref={ref} rotation={[.05,-.2,-.12]}>
  <mesh geometry={geometry}>{material}</mesh>
  <mesh position={[-.53,.74,-.06]} scale={[.54,.44,.57]} rotation-z={-.4}><sphereGeometry args={[1,40,32]} />{material}</mesh>
  <mesh position={[.51,.71,-.16]} scale={[.5,.4,.53]}><sphereGeometry args={[1,40,32]} />{material}</mesh>
  <Tube points={[[.04,.65,-.1],[.03,1.19,-.08],[.12,1.63,-.14],[-.12,1.89,-.13],[-.49,1.74,-.13],[-.62,1.44,-.22]]} radius={.19} color={colors['heart-artery']} />
  <Tube points={[[-.32,.59,.27],[-.25,1.08,.26],[-.3,1.37,.23],[-.69,1.53,.13]]} radius={.16} color={colors['pink']} />
  <Tube points={[[.62,.61,-.22],[.63,1.06,-.25],[.61,1.48,-.29]]} radius={.14} color={colors['heart-body']} />
  {([[-.06,1.8,-.12],[.14,1.72,-.13],[.3,1.55,-.1]] as Point[]).map((p,i) => <Tube key={i} points={[p,[p[0]-.02,p[1]+.35,p[2]-.02]]} radius={.055} color={colors['heart-artery']} />)}
  {(Object.keys(paths) as Vessel[]).map(v => { const color=colors[`risk-${riskLevel(risks[v])}`]; return <group key={v}><Tube points={paths[v]} radius={.027} color={color} glow={1.35} onClick={() => onSelect(v)} />
   {paths[v].slice(1,5).map((p,i) => { const side=v==='LAD'?1:-1; const end: Point=[p[0]+side*(.23+i*.035),p[1]-.23,p[2]-.07]; return <Tube key={i} points={[p,[p[0]+side*.1,p[1]-.12,p[2]+.025],end]} radius={.009} color={color} glow={.9} onClick={() => onSelect(v)} />; })}
  </group>; })}
  {Array.from({length:10},(_,i) => { const y=.4-i*.12; const x=.5+i*.017; const z=Math.sqrt(Math.max(0,1-y*y-x*x))*.85; return <Tube key={i} points={[[x,y,z],[x+.15,y-.1,z-.03],[x+.22,y-.27,z-.13]]} radius={.007} color={colors['heart-highlight']} glow={.12} />; })}
 </group>;
}
function Torso({ color }: { color: string }) {
 const mat = <meshStandardMaterial color={color} transparent opacity={.1} roughness={.85} depthWrite={false} />;
 return <group position={[0,.05,-.45]}>
  {[-1,1].map(s => <mesh key={s} position={[s*1.95,.1,-.1]} scale={[.95,1.7,.75]}><sphereGeometry args={[1,36,28]} />{mat}</mesh>)}
  {Array.from({length:7},(_,i) => <mesh key={i} position={[0,1.45-i*.42,0]} rotation-x={Math.PI/2} scale={[1,.8,1]}><torusGeometry args={[2.55-Math.abs(i-2)*.12,.022,8,64]} /><meshStandardMaterial color={color} transparent opacity={.22} /></mesh>)}
  <mesh position={[0,.1,1.95]}><boxGeometry args={[.22,2.5,.06]} />{mat}</mesh>
  <mesh position={[0,.1,-1.9]}><cylinderGeometry args={[.12,.12,3.4,12]} />{mat}</mesh>
 </group>;
}
export function HeartCanvas({ risks, onSelect }: { risks: Record<Vessel,number>; onSelect:(v:Vessel)=>void }) {
 const [colors,setColors]=useState<Palette | null>(null); const [reduced,setReduced]=useState(false); const [auto,setAuto]=useState(false); const [torso,setTorso]=useState(false);
 const controls=useRef<OrbitControlsType>(null); const container=useRef<HTMLDivElement>(null);
 useEffect(() => { const style=getComputedStyle(document.documentElement); const canvas=document.createElement('canvas'); canvas.width=canvas.height=1; const ctx=canvas.getContext('2d'); if(!ctx)return; const result = Object.fromEntries(colorTokens.map(token => {ctx.clearRect(0,0,1,1); ctx.fillStyle=style.getPropertyValue(`--${token}`).trim();ctx.fillRect(0,0,1,1);const pixel=ctx.getImageData(0,0,1,1).data; return [token,`#${Array.from(pixel).slice(0,3).map(v=>v.toString(16).padStart(2,'0')).join('')}`];})) as Palette; setColors(result);setReduced(matchMedia('(prefers-reduced-motion: reduce)').matches); },[]);
 return <><div ref={container} className="canvas-area" aria-label="Interactive 3D heart with coronary risk vessels">{colors && <Canvas dpr={1.25} camera={{ position:[0,.25,6.4], fov:40 }} gl={{antialias:true,alpha:true}}>
  <ambientLight intensity={.65} /><directionalLight position={[3,4,5]} intensity={3} color={colors['heart-highlight']} /><pointLight position={[-3,1,2]} intensity={25} color={colors.pink} /><pointLight position={[2,-1,2]} intensity={12} color={colors['heart-artery']} />
  <Environment resolution={64}><Lightformer position={[0,4,3]} scale={[6,4,1]} intensity={2} /><Lightformer position={[-3,0,1]} rotation-y={Math.PI/2} scale={[5,5,1]} color={colors.pink} intensity={1.5} /></Environment>
  <Suspense fallback={null}>{torso && <Torso color={colors.foreground} />}<Heart colors={colors} risks={risks} onSelect={onSelect} reduced={reduced} /></Suspense>
  <OrbitControls ref={controls} enablePan={false} minDistance={4} maxDistance={9} target={[0,.15,0]} autoRotate={auto && !reduced} autoRotateSpeed={.7} />
  <EffectComposer multisampling={0}><Bloom luminanceThreshold={1.1} intensity={.65} mipmapBlur /></EffectComposer>
 </Canvas>}</div><div className="viewer-tools">
 <Button variant="ghost" size="icon" aria-label="Toggle torso context" title="Torso context (lungs, ribs)" aria-pressed={torso} onClick={()=>setTorso(!torso)}><Layers /></Button>
 <Button variant="ghost" size="icon" aria-label="Toggle heart rotation" title="Auto-rotate" aria-pressed={auto} onClick={()=>setAuto(!auto)}><Rotate3D /></Button>
 <Button variant="ghost" size="icon" aria-label="Zoom in" title="Zoom in" onClick={()=>{controls.current?.dollyIn(1.2);controls.current?.update();}}><Plus /></Button>
 <Button variant="ghost" size="icon" aria-label="Zoom out" title="Zoom out" onClick={()=>{controls.current?.dollyOut(1.2);controls.current?.update();}}><Minus /></Button>
 <Button variant="ghost" size="icon" aria-label="Reset heart view" title="Reset view" onClick={()=>controls.current?.reset()}><RotateCcw /></Button>
 <Button variant="ghost" size="icon" aria-label="Expand heart view" title="Full screen" onClick={()=>{if(document.fullscreenElement)void document.exitFullscreen();else void container.current?.parentElement?.requestFullscreen();}}><Maximize /></Button>
 </div></>;
}
