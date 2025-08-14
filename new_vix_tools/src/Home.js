import React, { useState, useEffect, useRef } from 'react';
import { motion, useScroll, useTransform, useInView, AnimatePresence } from 'framer-motion';
import { 
  Database, Shield, Brain, GitBranch, FileSearch, BarChart3, 
  Network, Lock, Cpu, Upload, Search, BookOpen,
  ChevronRight, ArrowRight, CheckCircle, AlertCircle, 
  Globe, Server, Mail, Activity,
  Grid3x3, Workflow, FileText, Key, Sparkles,
  TrendingUp, Layers, Zap, Command, Eye
} from 'lucide-react';

// Animated Counter Component
const AnimatedCounter = ({ value, duration = 2000 }) => {
  const [count, setCount] = useState(0);
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });
  
  useEffect(() => {
    if (!isInView) return;
    
    const numericValue = parseInt(value.replace(/[^0-9]/g, ''));
    if (isNaN(numericValue)) {
      setCount(value);
      return;
    }
    
    const increment = numericValue / (duration / 16);
    let current = 0;
    
    const timer = setInterval(() => {
      current += increment;
      if (current >= numericValue) {
        setCount(value);
        clearInterval(timer);
      } else {
        const prefix = value.match(/^[^0-9]*/)[0];
        const suffix = value.match(/[^0-9]*$/)[0];
        setCount(`${prefix}${Math.floor(current).toLocaleString()}${suffix}`);
      }
    }, 16);
    
    return () => clearInterval(timer);
  }, [value, duration, isInView]);
  
  return <span ref={ref}>{count}</span>;
};

// Fixed Typewriter Effect Component
const TypewriterText = ({ text, delay = 0, onComplete }) => {
  const [displayedText, setDisplayedText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  
  useEffect(() => {
    const timeout = setTimeout(() => {
      setIsTyping(true);
      let index = 0;
      const interval = setInterval(() => {
        if (index <= text.length) {
          setDisplayedText(text.slice(0, index));
          index++;
        } else {
          setIsTyping(false);
          setIsComplete(true);
          clearInterval(interval);
          if (onComplete) onComplete();
        }
      }, 50);
      
      return () => clearInterval(interval);
    }, delay);
    
    return () => clearTimeout(timeout);
  }, [text, delay, onComplete]);
  
  return (
    <span className="inline-flex items-baseline">
      <span>{displayedText}</span>
      <AnimatePresence>
        {isTyping && !isComplete && (
          <motion.span
            key="cursor"
            initial={{ opacity: 1 }}
            animate={{ opacity: [1, 1, 0] }}
            exit={{ opacity: 0 }}
            transition={{ 
              duration: 0.8, 
              repeat: Infinity, 
              repeatType: "reverse",
              times: [0, 0.5, 1]
            }}
            className="inline-block w-[2px] h-8 bg-primary-red ml-1 align-middle"
            style={{ position: 'relative', top: '-0.1em' }}
          />
        )}
      </AnimatePresence>
    </span>
  );
};

// Spotlight Card Component
const SpotlightCard = ({ children, className = "" }) => {
  const divRef = useRef(null);
  const [isFocused, setIsFocused] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [opacity, setOpacity] = useState(0);

  const handleMouseMove = (e) => {
    if (!divRef.current || isFocused) return;
    const div = divRef.current;
    const rect = div.getBoundingClientRect();
    setPosition({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  };

  const handleFocus = () => {
    setIsFocused(true);
    setOpacity(1);
  };

  const handleBlur = () => {
    setIsFocused(false);
    setOpacity(0);
  };

  const handleMouseEnter = () => {
    setOpacity(1);
  };

  const handleMouseLeave = () => {
    setOpacity(0);
  };

  return (
    <div
      ref={divRef}
      onMouseMove={handleMouseMove}
      onFocus={handleFocus}
      onBlur={handleBlur}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className={`relative overflow-hidden bg-white border border-gray-200 ${className}`}
    >
      <div
        className="pointer-events-none absolute -inset-px opacity-0 transition duration-300"
        style={{
          opacity,
          background: `radial-gradient(600px circle at ${position.x}px ${position.y}px, rgba(239, 68, 68, 0.1), transparent 40%)`,
        }}
      />
      {children}
    </div>
  );
};

// Moving Border Card Component
const MovingBorderCard = ({ children, className = "", duration = 2000 }) => {
  return (
    <div className={`relative p-[1px] overflow-hidden ${className}`}>
      <motion.div
        className="absolute inset-0"
        style={{
          background: 'linear-gradient(90deg, transparent, #ef4444, transparent)',
        }}
        animate={{
          x: ['0%', '100%', '0%'],
        }}
        transition={{
          duration: duration / 1000,
          repeat: Infinity,
          ease: "linear",
        }}
      />
      <div className="relative bg-white h-full">
        {children}
      </div>
    </div>
  );
};

// Grid Background Component
const GridBackground = () => {
  return (
    <div className="absolute inset-0 overflow-hidden">
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-white" />
        <svg
          className="absolute inset-0 h-full w-full"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <pattern
              id="grid-pattern"
              width="32"
              height="32"
              patternUnits="userSpaceOnUse"
            >
              <path
                d="M0 32V0h32"
                fill="none"
                stroke="rgba(0, 0, 0, 0.04)"
                strokeWidth="1"
              />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid-pattern)" />
        </svg>
        <div className="absolute inset-0 bg-gradient-to-b from-white via-transparent to-white" />
      </div>
      <motion.div
        className="absolute -top-4 -left-4 w-72 h-72 bg-primary-red/5 rounded-full filter blur-3xl"
        animate={{
          x: [0, 100, 0],
          y: [0, -100, 0],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: "linear",
        }}
      />
      <motion.div
        className="absolute -bottom-4 -right-4 w-72 h-72 bg-primary-red/5 rounded-full filter blur-3xl"
        animate={{
          x: [0, -100, 0],
          y: [0, 100, 0],
        }}
        transition={{
          duration: 15,
          repeat: Infinity,
          ease: "linear",
        }}
      />
    </div>
  );
};

// Fixed Hero Section
const HeroSection = () => {
  const { scrollY } = useScroll();
  const y = useTransform(scrollY, [0, 300], [0, 50]);
  const opacity = useTransform(scrollY, [0, 300], [1, 0]);
  const [typingComplete, setTypingComplete] = useState(false);

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden bg-white">
      <GridBackground />
      
      <motion.div 
        style={{ y, opacity }}
        className="container mx-auto px-6 relative z-10"
      >
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center max-w-5xl mx-auto"
        >
          <motion.div
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="inline-flex items-center gap-2 px-6 py-2 mb-8"
          >
            <div className="flex items-center gap-2 px-4 py-2 bg-primary-red/5 border border-primary-red/20">
              <Sparkles className="w-4 h-4 text-primary-red" />
              <span className="text-sm font-semibold text-primary-red tracking-wider">
                AI-POWERED INTELLIGENCE
              </span>
            </div>
          </motion.div>
          
          <h1 className="text-5xl md:text-7xl font-bold text-darker-gray mb-6">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="block"
            >
              <TypewriterText 
                text="Hypergraph Platform" 
                delay={500} 
                onComplete={() => setTypingComplete(true)}
              />
            </motion.div>
            <motion.span
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 1.5 }}
              className="block text-primary-red mt-2"
            >
              Non-Financial Risk Intelligence
            </motion.span>
          </h1>
          
          <AnimatePresence>
            {typingComplete && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.8 }}
              >
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.2 }}
                  className="text-xl text-dark-gray mb-12 max-w-3xl mx-auto leading-relaxed"
                >
                  Enterprise-grade platform leveraging advanced AI and hypergraph architecture 
                  to transform risk management through intelligent data consolidation and multi-dimensional analysis.
                </motion.p>
                
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                  className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-4xl mx-auto mb-12"
                >
                  {[
                    { icon: Database, label: 'Unified Platform', value: '6', suffix: ' Datasets' },
                    { icon: Brain, label: 'AI Processing', value: '100', suffix: '% Local' },
                    { icon: Shield, label: 'Security Level', value: '100', suffix: '% On-premise' },
                    { icon: Network, label: 'Graph Nodes', value: '500', suffix: 'K+' }
                  ].map((item, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.5 + index * 0.1 }}
                      whileHover={{ y: -5 }}
                    >
                      <SpotlightCard className="p-6 h-full">
                        <item.icon className="w-10 h-10 text-primary-red mb-3" />
                        <div className="text-2xl font-bold text-darker-gray mb-1">
                          <AnimatedCounter value={item.value} />
                          <span className="text-lg">{item.suffix}</span>
                        </div>
                        <div className="text-sm text-dark-gray">{item.label}</div>
                      </SpotlightCard>
                    </motion.div>
                  ))}
                </motion.div>

                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.9 }}
                  className="flex flex-col sm:flex-row gap-4 justify-center"
                >
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="px-8 py-4 bg-primary-red text-white font-semibold flex items-center justify-center gap-2 hover:bg-dark-red transition-colors"
                  >
                    Explore Platform
                    <ArrowRight className="w-5 h-5" />
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="px-8 py-4 bg-white text-primary-red font-semibold border-2 border-primary-red flex items-center justify-center gap-2 hover:bg-primary-red/5 transition-colors"
                  >
                    <Eye className="w-5 h-5" />
                    View Documentation
                  </motion.button>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
          
          <AnimatePresence>
            {typingComplete && (
              <motion.div
                animate={{ y: [0, 10, 0] }}
                transition={{ repeat: Infinity, duration: 2 }}
                className="absolute bottom-10 left-1/2 transform -translate-x-1/2"
              >
                <ChevronRight className="w-8 h-8 text-primary-red/40 rotate-90" />
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </motion.div>
    </section>
  );
};

// About Section with Parallax
const AboutSection = () => {
  const { scrollY } = useScroll();
  const y1 = useTransform(scrollY, [0, 1000], [0, -50]);
  const y2 = useTransform(scrollY, [0, 1000], [0, -100]);

  const features = [
    {
      icon: GitBranch,
      title: 'Hypergraph Architecture',
      description: 'Multi-dimensional graph system enabling complex relationships and intelligent reasoning across interconnected datasets.',
      stats: '10x faster queries'
    },
    {
      icon: Brain,
      title: 'AI Enrichment Engine',
      description: 'Automated data enrichment with failure analysis, causal inference, and intelligent taxonomy mapping using state-of-the-art models.',
      stats: '99.9% accuracy'
    },
    {
      icon: Shield,
      title: 'Enterprise Security',
      description: 'Complete on-premise deployment with military-grade encryption, role-based access control, and zero data exposure.',
      stats: 'Zero breaches'
    },
    {
      icon: Database,
      title: 'Unified Integration',
      description: 'Seamlessly aggregate multiple risk datasets into a single, intelligent platform with real-time synchronization.',
      stats: '6 data sources'
    }
  ];

  return (
    <section className="py-24 bg-light-gray relative overflow-hidden">
      <motion.div style={{ y: y1 }} className="absolute inset-0 opacity-5">
        <div className="absolute top-20 left-20 w-64 h-64 bg-primary-red rounded-full filter blur-3xl" />
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-primary-red rounded-full filter blur-3xl" />
      </motion.div>
      
      <div className="container mx-auto px-6 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-6xl mx-auto"
        >
          <div className="text-center mb-16">
            <motion.div
              initial={{ width: 0 }}
              whileInView={{ width: '100px' }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="h-1 bg-primary-red mx-auto mb-8"
            />
            <motion.h2 
              style={{ y: y2 }}
              className="text-4xl md:text-5xl font-bold text-darker-gray mb-4"
            >
              Next-Generation Risk Intelligence
            </motion.h2>
            <p className="text-xl text-dark-gray max-w-3xl mx-auto">
              Transform your risk management with cutting-edge AI technology and advanced hypergraph analytics
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: index % 2 === 0 ? -30 : 30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.15, duration: 0.6 }}
              >
                <MovingBorderCard duration={3000} className="h-full">
                  <div className="p-8">
                    <div className="flex items-start gap-4">
                      <motion.div 
                        whileHover={{ rotate: 360 }}
                        transition={{ duration: 0.5 }}
                        className="w-14 h-14 bg-primary-red/10 flex items-center justify-center flex-shrink-0"
                      >
                        <feature.icon className="w-7 h-7 text-primary-red" />
                      </motion.div>
                      <div className="flex-1">
                        <h3 className="text-2xl font-bold text-darker-gray mb-3">{feature.title}</h3>
                        <p className="text-dark-gray mb-4 leading-relaxed">{feature.description}</p>
                        <div className="inline-flex items-center gap-2 px-3 py-1 bg-primary-red/5 text-primary-red">
                          <TrendingUp className="w-4 h-4" />
                          <span className="text-sm font-semibold">{feature.stats}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </MovingBorderCard>
              </motion.div>
            ))}
          </div>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mt-16 p-8 bg-white border-l-4 border-primary-red shadow-lg"
          >
            <div className="flex items-center gap-4 mb-4">
              <Command className="w-8 h-8 text-primary-red" />
              <p className="text-2xl text-darker-gray font-bold">Managed by NFR Quantitative Solutions</p>
            </div>
            <p className="text-dark-gray text-lg leading-relaxed">
              Enterprise-grade platform with dedicated support, continuous innovation, and deep expertise 
              in non-financial risk management. Trusted by leading financial institutions worldwide.
            </p>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};

// Dataset Stats with Counter Animations
const DatasetStats = () => {
  const datasets = [
    { name: 'External Loss Data', count: '7000', icon: Globe, color: 'from-blue-500 to-blue-600', prefix: '~' },
    { name: 'Internal Loss Data', count: '500000', icon: Database, color: 'from-green-500 to-green-600', prefix: '>' },
    { name: 'Controls', count: '4000', icon: Shield, color: 'from-purple-500 to-purple-600', suffix: '+' },
    { name: 'Issues', count: '6000', icon: AlertCircle, color: 'from-orange-500 to-orange-600', suffix: '+' },
    { name: 'Policies', count: '60', icon: FileText, color: 'from-indigo-500 to-indigo-600' },
    { name: 'RCSA', count: '2025.02', icon: Activity, color: 'from-pink-500 to-pink-600' }
  ];

  return (
    <section className="py-24 bg-white relative overflow-hidden">
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-primary-red/5 via-transparent to-primary-red/5" />
      </div>
      
      <div className="container mx-auto px-6 relative z-10">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <motion.div
              initial={{ scale: 0 }}
              whileInView={{ scale: 1 }}
              viewport={{ once: true }}
              className="inline-flex items-center gap-2 px-4 py-2 bg-primary-red/10 text-primary-red mb-8 border border-primary-red/20"
            >
              <Database className="w-4 h-4" />
              <span className="text-sm font-semibold tracking-wider">COMPREHENSIVE COVERAGE</span>
            </motion.div>
            <h2 className="text-4xl md:text-5xl font-bold text-darker-gray mb-4">
              Enterprise Dataset Integration
            </h2>
            <p className="text-xl text-dark-gray">
              AI-enriched data with intelligent taxonomy mapping and real-time updates
            </p>
          </motion.div>
          
          <div className="grid md:grid-cols-3 gap-8">
            {datasets.map((dataset, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, duration: 0.5 }}
              >
                <SpotlightCard className="h-full">
                  <div className="p-8">
                    <div className="flex items-start justify-between mb-6">
                      <motion.div
                        whileHover={{ scale: 1.1, rotate: 5 }}
                        className={`w-12 h-12 bg-gradient-to-br ${dataset.color} rounded-none flex items-center justify-center shadow-lg`}
                      >
                        <dataset.icon className="w-6 h-6 text-white" />
                      </motion.div>
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
                        className="w-20 h-20 border-2 border-dashed border-gray-200"
                      />
                    </div>
                    <h3 className="text-3xl font-bold text-darker-gray mb-2">
                      {dataset.prefix && <span>{dataset.prefix}</span>}
                      <AnimatedCounter value={dataset.count} />
                      {dataset.suffix && <span>{dataset.suffix}</span>}
                    </h3>
                    <p className="text-dark-gray font-medium mb-4">{dataset.name}</p>
                    <div className="flex items-center gap-2 text-sm text-gray-500">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span>AI Enriched</span>
                    </div>
                  </div>
                </SpotlightCard>
              </motion.div>
            ))}
          </div>
          
        </div>
      </div>
    </section>
  );
};

// Core Features with Advanced Cards
const CoreFeatures = () => {
  const features = [
    {
      icon: Grid3x3,
      title: 'Interactive Dashboard',
      description: 'Advanced analytics interface with real-time insights and customizable views.',
      capabilities: ['Real-time updates', 'Custom widgets', 'AI insights'],
      gradient: 'from-blue-500 to-cyan-500'
    },
    {
      icon: Upload,
      title: 'Task Automation',
      description: 'Intelligent task processing with asynchronous execution and tracking.',
      capabilities: ['10+ templates', 'Batch processing', 'Task scheduling'],
      gradient: 'from-purple-500 to-pink-500'
    },
    {
      icon: Network,
      title: 'Hypergraph Engine',
      description: 'Dynamic graph generation with role-based access and intelligent merging.',
      capabilities: ['Dynamic merging', 'Access control', 'Graph analytics'],
      gradient: 'from-green-500 to-emerald-500'
    },
    {
      icon: Workflow,
      title: 'Multi-Hop Analysis',
      description: 'Complex cross-dataset queries with intelligent relationship mapping.',
      capabilities: ['Cross-dataset', 'Pattern detection', 'Risk correlation'],
      gradient: 'from-orange-500 to-red-500'
    },
    {
      icon: Search,
      title: 'Natural Language',
      description: 'Query datasets using natural language with automatic visualization.',
      capabilities: ['NLP queries', 'Auto-viz', 'Export options'],
      gradient: 'from-indigo-500 to-purple-500'
    },
    {
      icon: BookOpen,
      title: 'Prompt Library',
      description: 'Collaborative prompt repository with version control and sharing.',
      capabilities: ['Templates', 'Versioning', 'Team sharing'],
      gradient: 'from-pink-500 to-rose-500'
    },
    {
      icon: BarChart3,
      title: 'Advanced Analytics',
      description: 'Specialized tools for deep analysis and pattern recognition.',
      capabilities: ['ML models', 'Clustering', 'Predictions'],
      gradient: 'from-teal-500 to-blue-500'
    },
    {
      icon: FileSearch,
      title: 'BYOD Support',
      description: 'Secure data upload with temporary merging and automatic cleanup.',
      capabilities: ['Secure upload', 'Auto-purge', 'Isolation'],
      gradient: 'from-amber-500 to-orange-500'
    },
    {
      icon: Server,
      title: 'Local Infrastructure',
      description: 'Complete on-premise deployment with open-source technology.',
      capabilities: ['On-premise', 'Open source', 'Full control'],
      gradient: 'from-gray-600 to-gray-800'
    }
  ];

  return (
    <section className="py-24 bg-light-gray">
      <div className="container mx-auto px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <motion.div
              initial={{ scale: 0 }}
              whileInView={{ scale: 1 }}
              viewport={{ once: true }}
              className="inline-flex items-center gap-2 px-4 py-2 bg-primary-red/10 text-primary-red mb-8 border border-primary-red/20"
            >
              <Zap className="w-4 h-4" />
              <span className="text-sm font-semibold tracking-wider">POWERFUL CAPABILITIES</span>
            </motion.div>
            <h2 className="text-4xl md:text-5xl font-bold text-darker-gray mb-4">
              Core Platform Features
            </h2>
            <p className="text-xl text-dark-gray max-w-3xl mx-auto">
              Comprehensive suite of tools designed for enterprise risk management excellence
            </p>
          </motion.div>
          
          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.05, duration: 0.5 }}
              >
                <motion.div
                  whileHover={{ y: -8, transition: { duration: 0.2 } }}
                  className="group h-full"
                >
                  <SpotlightCard className="h-full transition-all duration-300 hover:shadow-2xl">
                    <div className="p-8">
                      <div className="flex items-center gap-4 mb-6">
                        <motion.div
                          whileHover={{ rotate: 360 }}
                          transition={{ duration: 0.5 }}
                          className={`w-14 h-14 bg-gradient-to-br ${feature.gradient} flex items-center justify-center shadow-lg`}
                        >
                          <feature.icon className="w-7 h-7 text-white" />
                        </motion.div>
                        <h3 className="text-xl font-bold text-darker-gray">{feature.title}</h3>
                      </div>
                      
                      <p className="text-dark-gray mb-6 leading-relaxed">{feature.description}</p>
                      
                      <div className="space-y-3">
                        {feature.capabilities.map((capability, capIndex) => (
                          <motion.div
                            key={capIndex}
                            initial={{ opacity: 0, x: -20 }}
                            whileInView={{ opacity: 1, x: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: 0.1 * capIndex }}
                            className="flex items-center gap-2"
                          >
                            <ChevronRight className="w-4 h-4 text-primary-red flex-shrink-0" />
                            <span className="text-sm text-dark-gray">{capability}</span>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  </SpotlightCard>
                </motion.div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

// Security Section with Modern Design
const SecuritySection = () => {
  const securityFeatures = [
    {
      icon: Lock,
      title: 'Complete On-Premise',
      description: 'Full infrastructure deployment within your secure environment with zero external dependencies.',
      highlight: 'Zero Cloud Exposure',
      stats: '100% Local'
    },
    {
      icon: Key,
      title: 'Granular Access Control',
      description: 'Role-based permissions at dataset level with automatic graph filtering and audit trails.',
      highlight: 'Military-Grade Security',
      stats: 'RBAC Enabled'
    },
    {
      icon: Shield,
      title: 'Data Isolation',
      description: 'Automatic exclusion of unauthorized nodes with complete data segregation.',
      highlight: 'Complete Isolation',
      stats: 'Zero Leakage'
    },
    {
      icon: Cpu,
      title: 'Local AI Processing',
      description: 'All ML and LLM processing on local infrastructure using open-source models.',
      highlight: 'Self-Contained',
      stats: 'No External APIs'
    }
  ];

  return (
    <section className="py-24 bg-white relative overflow-hidden">
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-gray-50 via-white to-gray-50" />
      </div>
      
      <div className="container mx-auto px-6 relative z-10">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <motion.div
              initial={{ scale: 0 }}
              whileInView={{ scale: 1 }}
              viewport={{ once: true }}
              className="inline-flex items-center gap-2 px-4 py-2 bg-green-500/20 text-green-600 mb-8 border border-green-500/30"
            >
              <Shield className="w-4 h-4" />
              <span className="text-sm font-semibold tracking-wider">ENTERPRISE SECURITY</span>
            </motion.div>
            <h2 className="text-4xl md:text-5xl font-bold mb-4 text-darker-gray">
              Uncompromising Security
            </h2>
            <p className="text-xl text-dark-gray max-w-3xl mx-auto">
              Built with security-first architecture ensuring complete data sovereignty and privacy
            </p>
          </motion.div>
          
          <div className="grid md:grid-cols-2 gap-8 mb-12">
            {securityFeatures.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, duration: 0.5 }}
              >
                <motion.div
                  whileHover={{ scale: 1.02 }}
                  className="bg-white p-8 border border-gray-200 hover:border-primary-red/50 transition-all duration-300 shadow-lg"
                >
                  <div className="flex items-start gap-4">
                    <motion.div
                      whileHover={{ rotate: 360 }}
                      transition={{ duration: 0.5 }}
                      className="w-14 h-14 bg-primary-red/10 flex items-center justify-center flex-shrink-0"
                    >
                      <feature.icon className="w-7 h-7 text-primary-red" />
                    </motion.div>
                    <div className="flex-1">
                      <h3 className="text-2xl font-bold mb-3 text-darker-gray">{feature.title}</h3>
                      <p className="text-dark-gray mb-4 leading-relaxed">{feature.description}</p>
                      <div className="flex items-center gap-4">
                        <span className="inline-block px-3 py-1 bg-primary-red/10 text-primary-red text-sm font-semibold">
                          {feature.highlight}
                        </span>
                        <span className="text-green-600 text-sm font-medium">
                          {feature.stats}
                        </span>
                      </div>
                    </div>
                  </div>
                </motion.div>
              </motion.div>
            ))}
          </div>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="bg-gradient-to-r from-primary-red to-dark-red p-8 text-center"
          >
            <Lock className="w-12 h-12 mx-auto mb-4 text-white" />
            <h3 className="text-2xl font-bold mb-3 text-white">Zero Third-Party Data Sharing</h3>
            <p className="text-gray-100 max-w-2xl mx-auto text-lg">
              Your data never leaves your infrastructure. Complete sovereignty with 
              full audit trails and compliance with the strictest regulatory requirements.
            </p>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

// Contact Section
const ContactSection = () => {
  const [hoveredEmail, setHoveredEmail] = useState(false);
  
  return (
    <section className="py-24 bg-white relative overflow-hidden">
      <GridBackground />
      
      <div className="container mx-auto px-6 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-4xl mx-auto text-center"
        >
          <motion.div
            initial={{ scale: 0 }}
            whileInView={{ scale: 1 }}
            viewport={{ once: true }}
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary-red/10 text-primary-red mb-8 border border-primary-red/20"
          >
            <Mail className="w-4 h-4" />
            <span className="text-sm font-semibold tracking-wider">GET IN TOUCH</span>
          </motion.div>
          
          <h2 className="text-4xl md:text-5xl font-bold text-darker-gray mb-8">
            Connect With Our Team
          </h2>
          <p className="text-xl text-dark-gray mb-12">
            For queries, support, or feedback regarding the Hypergraph Platform
          </p>
          
          <motion.div
            whileHover={{ scale: 1.02 }}
            onHoverStart={() => setHoveredEmail(true)}
            onHoverEnd={() => setHoveredEmail(false)}
            className="inline-block"
          >
            <a
              href="mailto:NFIContractorSolution@redcubeus.com"
              className="inline-flex items-center gap-3 px-10 py-5 bg-primary-red hover:bg-dark-red text-white font-semibold text-lg transition-all duration-300 shadow-xl hover:shadow-2xl"
            >
              <Mail className={`w-6 h-6 ${hoveredEmail ? 'animate-pulse' : ''}`} />
              <span>NFIContractorSolution@redcubeus.com</span>
              <ArrowRight className={`w-5 h-5 transition-transform ${hoveredEmail ? 'translate-x-2' : ''}`} />
            </a>
          </motion.div>
          
          <div className="mt-20 pt-20 border-t border-gray-200">
            <div className="flex flex-col items-center gap-4">
              <div className="flex items-center gap-2">
                <Brain className="w-8 h-8 text-primary-red" />
                <span className="text-2xl font-bold text-darker-gray">Hypergraph Platform</span>
              </div>
              <p className="text-dark-gray">
                Managed by NFR Quantitative Solutions
              </p>
              <p className="text-gray-500 text-sm">
                © 2025 All rights reserved. Enterprise Risk Intelligence Platform.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};


// Main Home Component
const Home = () => {
  return (
    <div className="min-h-screen bg-white">
      <HeroSection />
      <AboutSection />
      <DatasetStats />
      <CoreFeatures />
      <SecuritySection />
      <ContactSection />
    </div>
  );
};

export default Home;