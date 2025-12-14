import React, { useState, useEffect, useRef } from 'react';
import { Terminal, Activity, Brain, Zap, Radio } from 'lucide-react';
import { useBrainSimulation } from './useBrainSimulation';
import BrainScene from './BrainScene';

function App() {
  const { topology, stats, lastSpikes, logs, isConnected, sendPrompt } = useBrainSimulation();
  const [input, setInput] = useState('');
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      sendPrompt(input);
      setInput('');
    }
  };

  return (
    <div className="w-full h-screen bg-deep-black text-white font-mono flex overflow-hidden">
      {/* 3D Visualization Area */}
      <div className="absolute inset-0 z-0">
        <BrainScene topology={topology} activeIndices={lastSpikes} />
      </div>

      {/* UI Overlay */}
      <div className="relative z-10 w-full h-full flex flex-col pointer-events-none">

        {/* Header */}
        <div className="w-full p-4 flex justify-between items-center bg-gradient-to-b from-black/80 to-transparent">
          <div className="flex items-center gap-2">
            <Brain className="text-royal-purple w-8 h-8" />
            <h1 className="text-2xl font-bold tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-royal-purple to-white">
              NEUROX-AI
            </h1>
            <span className={`text-xs px-2 py-0.5 rounded ${isConnected ? 'bg-green-900 text-green-200' : 'bg-red-900 text-red-200'}`}>
              {isConnected ? 'SYSTEM ONLINE' : 'DISCONNECTED'}
            </span>
          </div>
          <div className="text-xs text-gray-400">
            v0.1.0 | GPU ACCELERATED | SPIKING NEURAL NETWORK
          </div>
        </div>

        {/* Main Content Area */}
        <div className="flex-1 flex justify-between p-6">

          {/* Left Panel: Statistics */}
          <div className="w-64 space-y-4 pointer-events-auto">
             <StatCard
               title="NEUROMODULATION"
               icon={<Zap className="w-4 h-4 text-yellow-400" />}
             >
               <StatRow label="Dopamine" value={stats?.dopamine || 0} color="bg-yellow-400" />
               <StatRow label="Serotonin" value={stats?.serotonin || 0} color="bg-pink-500" />
               <StatRow label="Norepinephrine" value={stats?.norepinephrine || 0} color="bg-red-500" />
             </StatCard>

             <StatCard
               title="OSCILLATIONS"
               icon={<Radio className="w-4 h-4 text-cyan-400" />}
             >
               <StatRow label="Theta (4-8Hz)" value={(stats?.theta || 0)/10} color="bg-cyan-400" />
               <StatRow label="Gamma (30Hz+)" value={(stats?.gamma || 0)/100} color="bg-blue-500" />
               <div className="flex justify-between text-xs mt-2 text-gray-400">
                 <span>Criticality:</span>
                 <span className="text-white">{(stats?.criticality || 0).toFixed(3)}</span>
               </div>
             </StatCard>

             <StatCard
               title="COGNITIVE STATE"
               icon={<Activity className="w-4 h-4 text-green-400" />}
             >
               <div className="text-xs mb-1 text-gray-400">Attention Level</div>
               <div className="w-full bg-gray-800 h-1 mb-3">
                 <div className="bg-green-400 h-1 transition-all duration-300" style={{ width: `${(stats?.attention || 0) * 100}%` }}></div>
               </div>

               <div className="text-xs mb-1 text-gray-400">Total Spikes</div>
               <div className="text-xl font-bold text-white">{lastSpikes.length}</div>
             </StatCard>
          </div>

          {/* Right Panel: Logs & Input */}
          <div className="w-96 flex flex-col gap-4 pointer-events-auto">
            {/* Log Window */}
            <div className="flex-1 bg-black/80 border border-royal-purple/30 rounded p-3 backdrop-blur-sm flex flex-col min-h-0">
               <div className="flex items-center gap-2 text-royal-purple mb-2 border-b border-royal-purple/20 pb-1">
                 <Terminal className="w-4 h-4" />
                 <span className="text-xs font-bold">SYSTEM LOG</span>
               </div>
               <div ref={logRef} className="flex-1 overflow-y-auto text-xs space-y-1 font-mono scrollbar-thin scrollbar-thumb-royal-purple/50">
                 {logs.length === 0 && <div className="text-gray-500 italic">Waiting for connection...</div>}
                 {logs.map((log, i) => (
                   <div key={i} className={`${log.startsWith('BRAIN >') ? 'text-green-400' : log.startsWith('USER >') ? 'text-blue-400' : 'text-gray-300'}`}>
                     {log}
                   </div>
                 ))}
               </div>
            </div>

            {/* Input Field */}
            <form onSubmit={handleSubmit} className="bg-black/80 border border-royal-purple/50 rounded p-2 flex gap-2 backdrop-blur-sm">
               <span className="text-royal-purple font-bold">{'>'}</span>
               <input
                 type="text"
                 value={input}
                 onChange={e => setInput(e.target.value)}
                 className="flex-1 bg-transparent border-none outline-none text-sm text-white placeholder-gray-600"
                 placeholder="Enter command..."
                 autoFocus
               />
            </form>
          </div>

        </div>
      </div>
    </div>
  );
}

function StatCard({ title, icon, children }: { title: string, icon: any, children: React.ReactNode }) {
  return (
    <div className="bg-black/80 border border-gray-800 rounded p-3 backdrop-blur-sm">
      <div className="flex items-center gap-2 text-gray-400 mb-3 text-xs font-bold tracking-widest border-b border-gray-800 pb-2">
        {icon}
        {title}
      </div>
      {children}
    </div>
  );
}

function StatRow({ label, value, color }: { label: string, value: number, color: string }) {
  return (
    <div className="mb-2">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-400">{label}</span>
        <span className="text-white font-mono">{value.toFixed(3)}</span>
      </div>
      <div className="w-full bg-gray-900 h-1.5 rounded-full overflow-hidden">
        <div
          className={`h-full ${color} transition-all duration-300 ease-out`}
          style={{ width: `${Math.min(value * 100, 100)}%` }}
        ></div>
      </div>
    </div>
  );
}

export default App;
