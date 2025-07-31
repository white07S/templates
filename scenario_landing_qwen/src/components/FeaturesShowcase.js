// src/components/FeaturesShowcase.js
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Database, Shield, BarChart3, FileSearch, GitBranch } from 'lucide-react';

const features = [
  {
    id: 'multi-agent',
    title: "Multi-Agent AI System",
    subtitle: "Intelligent Risk Analysis",
    description: "Specialized AI agents work in concert to deliver comprehensive risk scenario analysis through advanced reasoning and validation processes.",
    color: "#e60000",
    icon: <Brain className="h-6 w-6" />,
    details: [
      { 
        name: "Data Analyst Agent", 
        role: "Historical loss event analysis",
        capabilities: "Pattern recognition, trend analysis, anomaly detection"
      },
      { 
        name: "Scenario Modeler Agent", 
        role: "Predictive scenario creation",
        capabilities: "Monte Carlo simulations, stress testing, impact modeling"
      },
      { 
        name: "Compliance Verification Agent", 
        role: "Regulatory alignment validation",
        capabilities: "Policy checking, regulatory mapping, compliance scoring"
      },
      { 
        name: "Control Environment Tester", 
        role: "Control failure analysis",
        capabilities: "Gap analysis, vulnerability assessment, mitigation strategies"
      }
    ],
    stats: {
      accuracy: "94%",
      processing: "<2min",
      coverage: "100%"
    }
  },
  {
    id: 'scenario-library',
    title: "Scenario Library Management",
    subtitle: "Comprehensive Risk Repository",
    description: "Enterprise-grade scenario management with multi-dimensional taxonomy tracking, version control, and cross-divisional sharing capabilities.",
    color: "#bd000c",
    icon: <Database className="h-6 w-6" />,
    features: [
      {
        title: "Taxonomy Management",
        items: ["Operational Risk", "Credit Risk", "Market Risk", "Liquidity Risk", "Strategic Risk"]
      },
      {
        title: "Version Control",
        items: ["Change tracking", "Audit trail", "Rollback capability", "Approval workflows"]
      },
      {
        title: "Advanced Search",
        items: ["Semantic search", "Filter by division", "Risk type filtering", "Impact-based sorting"]
      }
    ],
    metrics: {
      scenarios: "1,284",
      divisions: "12",
      updates: "Daily"
    }
  },
  {
    id: 'local-ai',
    title: "Local AI Infrastructure",
    subtitle: "Data Sovereignty Guaranteed",
    description: "Complete on-premise AI processing using state-of-the-art open-source models, ensuring data never leaves UBS infrastructure.",
    color: "#4d3c2f",
    icon: <Shield className="h-6 w-6" />,
    infrastructure: [
      {
        model: "Qwen-72B",
        purpose: "Primary analysis engine",
        performance: "Superior reasoning capabilities",
        status: "Production"
      },
      {
        model: "Mistral-8x7B",
        purpose: "Scenario generation",
        performance: "High-speed inference",
        status: "Production"
      },
      {
        model: "Custom Risk Models",
        purpose: "UBS-specific analysis",
        performance: "Domain-optimized",
        status: "Beta"
      }
    ],
    security: [
      "Zero external API calls",
      "End-to-end encryption",
      "Role-based access control",
      "Audit logging"
    ]
  }
];

const FeaturesShowcase = () => {
  const [activeFeature, setActiveFeature] = useState('multi-agent');
  const activeFeatureData = features.find(f => f.id === activeFeature);

  return (
    <section className="py-20 bg-gradient-to-b from-[#fafafa] to-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-bold text-[#1a1a1a] mb-4">
            Platform Capabilities
          </h2>
          <p className="text-lg text-[#666666] max-w-3xl mx-auto">
            Enterprise-grade risk management powered by cutting-edge AI technology
          </p>
        </motion.div>
        
        <div className="grid lg:grid-cols-3 gap-6 mb-12">
          {features.map((feature, index) => (
            <motion.div
              key={feature.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              whileHover={{ y: -8 }}
              className={`group relative bg-white rounded-xl border overflow-hidden cursor-pointer transition-all duration-300 ${
                activeFeature === feature.id 
                  ? 'border-[#e60000] shadow-xl' 
                  : 'border-[#f0f0f0] hover:border-[#e60000]/50 hover:shadow-lg'
              }`}
              onClick={() => setActiveFeature(feature.id)}
            >
              <div className="absolute inset-0 bg-gradient-to-br from-[#e60000]/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              
              <div className="relative p-6">
                <div className="flex items-start justify-between mb-4">
                  <div 
                    className="p-3 rounded-lg transition-colors" 
                    style={{ 
                      backgroundColor: activeFeature === feature.id ? `${feature.color}20` : '#f5f5f5',
                      color: feature.color 
                    }}
                  >
                    {feature.icon}
                  </div>
                  {activeFeature === feature.id && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="w-2 h-2 rounded-full bg-[#e60000]"
                    />
                  )}
                </div>
                
                <h3 className="text-lg font-bold text-[#1a1a1a] mb-1 group-hover:text-[#e60000] transition-colors">
                  {feature.title}
                </h3>
                <p className="text-sm text-[#7b6b59] mb-3">{feature.subtitle}</p>
                <p className="text-sm text-[#666666] line-clamp-2">{feature.description}</p>
                
                {feature.stats && (
                  <div className="flex gap-4 mt-4 pt-4 border-t border-[#f5f5f5]">
                    {Object.entries(feature.stats).map(([key, value]) => (
                      <div key={key} className="text-center">
                        <p className="text-lg font-bold text-[#e60000]">{value}</p>
                        <p className="text-xs text-[#7b6b59] capitalize">{key}</p>
                      </div>
                    ))}
                  </div>
                )}
                
                {feature.metrics && (
                  <div className="flex gap-4 mt-4 pt-4 border-t border-[#f5f5f5]">
                    {Object.entries(feature.metrics).map(([key, value]) => (
                      <div key={key} className="text-center">
                        <p className="text-lg font-bold text-[#1a1a1a]">{value}</p>
                        <p className="text-xs text-[#7b6b59] capitalize">{key}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </div>
          
        <AnimatePresence mode="wait">
          {activeFeatureData && (
            <motion.div
              key={activeFeature}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
              className="bg-white rounded-2xl shadow-xl overflow-hidden"
            >
              <div className="grid lg:grid-cols-2 gap-0">
                <div className="p-8 lg:p-12 flex flex-col h-full">
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 }}
                    className="flex flex-col h-full"
                  >
                    <div className="flex items-center gap-4 mb-6">
                      <div 
                        className="p-4 rounded-xl" 
                        style={{ backgroundColor: `${activeFeatureData.color}15`, color: activeFeatureData.color }}
                      >
                        {activeFeatureData.icon}
                      </div>
                      <div>
                        <h3 className="text-2xl font-bold text-[#1a1a1a]">
                          {activeFeatureData.title}
                        </h3>
                        <p className="text-sm text-[#7b6b59]">{activeFeatureData.subtitle}</p>
                      </div>
                    </div>
                    
                    <p className="text-lg text-[#666666] mb-8 leading-relaxed">
                      {activeFeatureData.description}
                    </p>
                    
                    <div className="flex-1 overflow-y-auto">
                      {activeFeatureData.details && (
                        <div className="space-y-3">
                          <h4 className="font-bold text-[#1a1a1a] mb-3">AI Agent Capabilities</h4>
                          <div className="grid gap-3">
                            {activeFeatureData.details.slice(0, 3).map((agent, idx) => (
                              <motion.div 
                                key={idx}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.3 + idx * 0.1 }}
                                className="bg-[#fafafa] rounded-lg p-3 border border-[#f0f0f0]"
                              >
                                <div className="flex items-start gap-3">
                                  <div className="w-7 h-7 rounded-full bg-[#e60000]/10 flex items-center justify-center flex-shrink-0">
                                    <span className="text-[#e60000] text-xs font-bold">{idx + 1}</span>
                                  </div>
                                  <div className="flex-1">
                                    <h5 className="font-semibold text-[#1a1a1a] text-sm">{agent.name}</h5>
                                    <p className="text-xs text-[#7b6b59] mt-0.5">{agent.role}</p>
                                  </div>
                                </div>
                              </motion.div>
                            ))}
                          </div>
                          {activeFeatureData.details.length > 3 && (
                            <p className="text-xs text-[#7b6b59] text-center pt-2">+{activeFeatureData.details.length - 3} more capabilities</p>
                          )}
                        </div>
                      )}
                    
                      {activeFeatureData.features && (
                        <div className="space-y-4">
                          {activeFeatureData.features.map((feature, idx) => (
                            <motion.div 
                              key={idx}
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              transition={{ delay: 0.3 + idx * 0.1 }}
                            >
                              <h4 className="font-bold text-[#1a1a1a] mb-2 text-sm">{feature.title}</h4>
                              <div className="grid grid-cols-2 gap-1.5">
                                {feature.items.slice(0, 4).map((item, itemIdx) => (
                                  <div key={itemIdx} className="flex items-center gap-2">
                                    <div className="w-1.5 h-1.5 rounded-full bg-[#e60000]" />
                                    <span className="text-xs text-[#666666]">{item}</span>
                                  </div>
                                ))}
                              </div>
                            </motion.div>
                          ))}
                        </div>
                      )}
                    </div>
                  </motion.div>
                </div>
                
                <div className="bg-gradient-to-br from-[#fafafa] to-[#f5f5f5] p-8 lg:p-12 border-l border-[#f0f0f0] flex flex-col h-full">
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.4 }}
                    className="flex flex-col h-full"
                  >
                    <div className="flex-1 flex flex-col">
                      {activeFeatureData.infrastructure && (
                        <div className="flex-1 flex flex-col">
                          <h4 className="font-bold text-[#1a1a1a] mb-4">Key Features</h4>
                          <div className="space-y-3 flex-1">
                            {activeFeatureData.infrastructure.slice(0, 2).map((model, idx) => (
                              <motion.div 
                                key={idx}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.5 + idx * 0.1 }}
                                className="bg-white rounded-lg p-3 border border-[#e6e6e6]"
                              >
                                <div className="flex items-start justify-between mb-1">
                                  <h5 className="font-semibold text-[#1a1a1a] text-sm">{model.model}</h5>
                                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                                    model.status === 'Production' 
                                      ? 'bg-[#4d3c2f]/10 text-[#4d3c2f]' 
                                      : 'bg-[#f2d88e]/30 text-[#7b6b59]'
                                  }`}>
                                    {model.status}
                                  </span>
                                </div>
                                <p className="text-xs text-[#7b6b59]">{model.purpose}</p>
                              </motion.div>
                            ))}
                          </div>
                          
                          {activeFeatureData.security && (
                            <div className="mt-4 p-3 bg-[#e60000]/5 rounded-lg border border-[#e60000]/10">
                              <h5 className="font-semibold text-[#e60000] mb-2 text-sm">Security Highlights</h5>
                              <div className="grid grid-cols-1 gap-1.5">
                                {activeFeatureData.security.slice(0, 3).map((item, idx) => (
                                  <div key={idx} className="flex items-center gap-2">
                                    <Shield className="h-3 w-3 text-[#e60000]" />
                                    <span className="text-xs text-[#666666]">{item}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </motion.div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
};

export default FeaturesShowcase;