import { motion } from 'framer-motion';
import { Cpu, Activity, ArrowRight, MessageSquare, Dna, Compass, GitMerge } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <div className="space-y-32 pb-20">
      {/* Hero Section */}
      <section className="relative pt-10">
        <div className="absolute -top-24 -left-24 w-96 h-96 bg-accent/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative space-y-8">
          <motion.h1
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-6xl md:text-8xl font-bold tracking-tight text-white leading-[0.9]"
          >
            Simulating the <br />
            <span className="text-accent">Architecture of Thought</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-xl text-text-secondary max-w-2xl leading-relaxed"
          >
            NeuroxAI is a hyper-realistic neuromorphic platform built in Rust.
            From ion-channel dynamics to high-level metacognition. We're rebuilding the brain in silicon.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="flex flex-wrap gap-4 pt-4"
          >
            <Link to="/docs/architecture/system_overview" className="h-12 px-8 rounded-lg bg-accent text-background font-bold hover:opacity-90 transition-all flex items-center gap-2">
              Explore Architecture <ArrowRight size={18} />
            </Link>
            <a href="https://github.com/TheRemyyy/neurox-ai" target="_blank" rel="noopener noreferrer" className="h-12 px-8 rounded-lg border border-border hover:border-accent/30 text-text-primary font-medium transition-all flex items-center gap-2">
              GitHub
            </a>
          </motion.div>
        </div>
      </section>

      {/* About / Mission Section */}
      <section className="py-20 border-y border-border/50 relative overflow-hidden">
        <div className="grid md:grid-cols-2 gap-16 items-center">
          <div className="space-y-6">
            <h2 className="text-3xl font-bold tracking-tight text-accent">Beyond matrix multiplication</h2>
            <div className="space-y-4 text-text-secondary text-lg leading-relaxed font-medium">
              <p>
                Most "AI" today is just linear algebra disguised as intelligence. <span className="font-semibold text-text-primary">NeuroxAI is different.</span> We strictly adhere to biological constraints because we believe that General Intelligence is an emergent property of complex, evolutionarily conserved systems.
              </p>
              <p>
                <span className="text-text-primary border-b-2 border-accent/30">210 hours of intensive development</span> have resulted in a platform that simulates distinct brain organs: the Cortex, Basal Ganglia, Hippocampus, and Cerebellum, all working in a synchronized temporal loop.
              </p>
              <p>
                By utilizing <span className="font-semibold text-text-primary">Triplet STDP</span> and <span className="font-semibold text-text-primary">Neuromodulation</span>, NeuroxAI doesn't just process data; it learns and adapts through simulated chemical states.
              </p>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <StatCard label="Accuracy" value="??.?%" sub="MNIST / 4-bit" />
            <StatCard label="Speedup" value="100x" sub="CUDA V1 Kernels" />
            <StatCard label="Neurons" value="1-10M" sub="Target Scale" />
            <StatCard label="Precision" value="0.1ms" sub="Simulation Step" />
          </div>
        </div>
      </section>

      {/* Key Modules */}
      <section className="space-y-12">
        <div>
          <h2 className="text-4xl font-bold tracking-tight">Biological systems</h2>
          <p className="text-text-secondary mt-2 max-w-xl">A complete cognitive stack, from hardware to consciousness.</p>
        </div>
        <ul className="divide-y divide-border">
          <ModuleRow icon={Cpu} title="Laminar Cortex" desc="6-layer microcircuitry implementing predictive coding. Minimizing free energy across 5 hierarchical levels." />
          <ModuleRow icon={Activity} title="Dopaminergic RL" desc="Basal ganglia loops for action selection. Real-time TD-learning modulated by dopamine bursts and pauses." />
          <ModuleRow icon={MessageSquare} title="Dual-Stream Language" desc="Ventral and dorsal pathways for sound-to-meaning and sound-to-articulation mapping. GPT-1.0 capability in SNN." />
          <ModuleRow icon={GitMerge} title="Synaptic Plasticity" desc="Triplet STDP, BCM theory, and homeostatic scaling. The network physically rewires itself based on activity." />
          <ModuleRow icon={Compass} title="Cognitive Maps" desc="Grid cells and place cells provide a metric coordinate system for both spatial navigation and abstract reasoning." />
          <ModuleRow icon={Dna} title="Active Dendrites" desc="NMDA plateau potentials turn single neurons into 2-layer networks, boosting computational capacity by 5x." />
        </ul>
      </section>

      {/* Philosophy Callout */}
      <section className="p-12 rounded-2xl bg-surface border border-border text-center space-y-8 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-accent/5 rounded-full blur-3xl" />
        <h2 className="text-3xl font-bold tracking-tight">The genome of intelligence</h2>
        <p className="text-xl text-text-secondary max-w-2xl mx-auto leading-relaxed">
          Intelligence is not a single algorithm. It is an orchestrated set of specialized systems, shaped by evolution.
        </p>
        <div className="pt-4 flex justify-center">
          <Link to="/docs/internals/data_training" className="flex items-center gap-2 text-accent font-semibold text-sm hover:underline">
            Read the Semantic Genome <ArrowRight size={16} />
          </Link>
        </div>
      </section>
    </div>
  );
}

function StatCard({ label, value, sub }: { label: string, value: string, sub: string }) {
  return (
    <div className="p-6 rounded-xl border border-border bg-surface/50 group hover:border-accent/40 transition-all">
      <div className="text-sm text-text-secondary tracking-wide mb-1">{label}</div>
      <div className="text-3xl font-bold text-white mb-1">{value}</div>
      <div className="text-xs text-accent font-mono">{sub}</div>
    </div>
  );
}

function ModuleRow({ icon: Icon, title, desc }: { icon: any, title: string, desc: string }) {
  return (
    <li className="flex gap-5 py-5 first:pt-0 group">
      <span className="shrink-0 mt-0.5 text-text-secondary group-hover:text-accent transition-colors">
        <Icon size={20} strokeWidth={1.75} />
      </span>
      <div className="min-w-0">
        <h3 className="text-base font-semibold text-white tracking-tight">{title}</h3>
        <p className="text-text-secondary text-sm leading-relaxed mt-1">
          {desc}
        </p>
      </div>
    </li>
  );
}
