import { useState, useEffect, useRef, useCallback } from 'react';

// Types matching Rust structs
export interface NeuronTopology {
  id: number;
  x: number;
  y: number;
  z: number;
  region: string;
  neuron_type: string;
}

export interface BrainStats {
  dopamine: number;
  serotonin: number;
  norepinephrine: number;
  theta: number;
  gamma: number;
  criticality: number;
  attention: number;
}

export interface UpdatePacket {
  type: 'update';
  tick: number;
  spikes: number[];
  stats: BrainStats;
}

export interface InitPacket {
  type: 'topology';
  topology: {
    neurons: NeuronTopology[];
  };
}

export function useBrainSimulation() {
  const [topology, setTopology] = useState<NeuronTopology[]>([]);
  const [stats, setStats] = useState<BrainStats | null>(null);
  const [lastSpikes, setLastSpikes] = useState<number[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    const ws = new WebSocket('ws://localhost:3000/ws');

    ws.onopen = () => {
      console.log('Connected to NeuroxAI Brain');
      setIsConnected(true);
      setLogs(prev => [...prev, '>>> CONNECTION ESTABLISHED']);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'topology') {
        const packet = data as InitPacket;
        console.log(`Received topology with ${packet.topology.neurons.length} neurons`);
        setTopology(packet.topology.neurons);
        setLogs(prev => [...prev, `>>> TOPOLOGY RECEIVED: ${packet.topology.neurons.length} NEURONS`]);
      } else if (data.type === 'update') {
        const packet = data as UpdatePacket;
        setStats(packet.stats);
        setLastSpikes(packet.spikes);
      } else if (data.type === 'response') {
        setLogs(prev => [...prev, `BRAIN > ${data.payload}`]);
      }
    };

    ws.onclose = () => {
      console.log('Disconnected');
      setIsConnected(false);
      setLogs(prev => [...prev, '>>> CONNECTION LOST']);
      // Reconnect after 3s
      setTimeout(connect, 3000);
    };

    socketRef.current = ws;
  }, []);

  const sendPrompt = useCallback((text: string) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      setLogs(prev => [...prev, `USER > ${text}`]);
      socketRef.current.send(JSON.stringify({
        type: 'prompt',
        payload: text
      }));
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      socketRef.current?.close();
    };
  }, [connect]);

  return { topology, stats, lastSpikes, logs, isConnected, sendPrompt };
}
