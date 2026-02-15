import { NavLink } from 'react-router-dom';
import { 
  Brain, Cpu, Zap, Github, X,
  Layers, Dna, Compass, TrendingUp, Lightbulb,
  GitMerge, FlaskConical, HardDrive, MessageSquare, Database,
  Terminal, Activity, Calculator
} from 'lucide-react';

const navItems = [
  { label: 'Introduction', path: '/', icon: Brain },
  { label: 'CLI Reference', path: '/docs/cli_reference.md', icon: Terminal },
  { label: 'Architecture', path: '/docs/architecture/system_overview.md', icon: Cpu },
  { 
    label: 'Modules', 
    items: [
      { label: 'Cortical Hierarchy', path: '/docs/modules/cortical_hierarchy.md', icon: Layers },
      { label: 'Subcortical Systems', path: '/docs/modules/subcortical_systems.md', icon: Dna },
      { label: 'Spatial Navigation', path: '/docs/modules/spatial_navigation.md', icon: Compass },
      { label: 'Neuron Models', path: '/docs/modules/neuron_models.md', icon: Zap },
      { label: 'Cognitive Upgrades', path: '/docs/modules/cognitive_upgrades.md', icon: TrendingUp },
      { label: 'Reasoning & Motivation', path: '/docs/modules/reasoning_motivation.md', icon: Lightbulb },
      { label: 'Problem Solving', path: '/docs/modules/problem_solving.md', icon: Calculator },
      { label: 'MNIST Benchmark', path: '/docs/modules/mnist_benchmark.md', icon: Activity },
    ]
  },
  {
    label: 'Internals',
    items: [
      { label: 'Plasticity', path: '/docs/internals/plasticity.md', icon: GitMerge },
      { label: 'Neuromodulation', path: '/docs/internals/neuromodulation.md', icon: FlaskConical },
      { label: 'GPU Acceleration', path: '/docs/internals/gpu_acceleration.md', icon: HardDrive },
      { label: 'Language & Cognition', path: '/docs/internals/language_cognition.md', icon: MessageSquare },
      { label: 'Data & Training', path: '/docs/internals/data_training.md', icon: Database },
    ]
  }
];

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export const Sidebar = ({ isOpen, onClose }: SidebarProps) => {
  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40 lg:hidden" 
          onClick={onClose}
        />
      )}

      <aside className={`
        fixed inset-y-0 left-0 z-50 w-64 border-r border-border bg-background transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        flex flex-col h-screen overflow-y-auto
      `}>
        <div className="p-6 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-accent rounded flex items-center justify-center text-background text-primary">
              <Brain size={20} strokeWidth={2.5} />
            </div>
            <span className="font-semibold tracking-tight text-lg">NeuroxAI</span>
          </div>
          <button onClick={onClose} className="lg:hidden text-text-secondary hover:text-text-primary">
            <X size={20} />
          </button>
        </div>
        
        <nav className="flex-1 p-4 space-y-2 text-sm">
          {navItems.map((group, i) => (
            <div key={i} className="space-y-2">
              {group.icon ? (
                <NavLink 
                  to={group.path}
                  onClick={onClose}
                  className={({ isActive }) => `flex items-center gap-3 px-3 py-2 rounded-md transition-colors ${isActive ? 'bg-surface text-accent font-medium' : 'text-text-secondary hover:text-text-primary'}`}
                >
                  <group.icon size={18} />
                  {group.label}
                </NavLink>
              ) : (
                <>
                  <h3 className="px-3 text-xs font-bold uppercase tracking-widest text-text-secondary opacity-50 mb-3">{group.label}</h3>
                  <div className="space-y-1">
                    {group.items?.map((item, j) => (
                      <NavLink 
                        key={j}
                        to={item.path}
                        onClick={onClose}
                        className={({ isActive }) => `flex items-center gap-3 px-3 py-1.5 rounded-md transition-colors border-l-2 ml-2 ${isActive ? 'border-accent text-accent bg-surface/50 font-medium' : 'border-transparent text-text-secondary hover:text-text-primary hover:border-border'}`}
                      >
                        {({ isActive }) => (
                          <>
                            {item.icon && <item.icon size={14} className={isActive ? 'text-accent' : 'text-text-secondary opacity-70'} />}
                            {item.label}
                          </>
                        )}
                      </NavLink>
                    ))}
                  </div>
                </>
              )}
            </div>
          ))}
        </nav>

        <div className="p-4 border-t border-border space-y-2">
          <a href="https://github.com/TheRemyyy/neurox-ai" target="_blank" rel="noopener noreferrer" className="flex items-center gap-3 px-3 py-2 text-text-secondary hover:text-text-primary transition-colors">
            <Github size={18} />
            GitHub Source
          </a>
        </div>
      </aside>
    </>
  );
};