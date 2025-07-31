import React from 'react';
import { motion } from 'framer-motion';
import { ChartBar, Cpu, Database, TrendingUp } from 'lucide-react';

const stats = [
  { 
    title: "Total Scenarios", 
    value: "1,284", 
    icon: <Database className="h-6 w-6" />,
    trend: "+18%",
    description: "Comprehensive risk library",
    details: [
      { label: "Active", value: "1,102" },
      { label: "In Review", value: "182" }
    ],
    className: "md:col-span-2 md:row-span-1"
  },
  { 
    title: "AI-Generated", 
    value: "892", 
    percentage: "69.4%",
    icon: <Cpu className="h-6 w-6" />,
    trend: "+32%",
    description: "ML-powered scenarios",
    details: [
      { label: "Human", value: "327" },
      { label: "Legacy", value: "65" }
    ],
    className: "md:col-span-1 md:row-span-1"
  },
  { 
    title: "By Taxonomy", 
    value: "1,284", 
    subtitle: "Total across categories",
    icon: <ChartBar className="h-6 w-6" />,
    description: "Risk distribution",
    categories: [
      { name: "Operational", value: "642", percentage: 50, color: "#e60000" },
      { name: "Credit", value: "318", percentage: 25, color: "#bd000c" },
      { name: "Market", value: "324", percentage: 25, color: "#4d3c2f" },
      { name: "Liquidity", value: "0", percentage: 0, color: "#7b6b59" }
    ],
    className: "md:col-span-2 md:row-span-1"
  },
  { 
    title: "Processing Status", 
    value: "3", 
    subtitle: "Active processes",
    icon: <TrendingUp className="h-6 w-6" />,
    description: "Real-time analysis",
    processMetrics: [
      { label: "Analyzing", value: "142", active: true },
      { label: "Validating", value: "89", active: true },
      { label: "Queued", value: "47", active: false }
    ],
    className: "md:col-span-1 md:row-span-1"
  }
];

const StatsDashboard = () => {
  return (
    <section className="py-20 bg-gradient-to-b from-white to-[#fafafa]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl md:text-4xl font-bold text-[#1a1a1a] mb-4">
            Risk Intelligence Metrics
          </h2>
          <p className="text-lg text-[#666666] max-w-3xl mx-auto">
            Real-time operational insights powered by advanced AI analysis
          </p>
        </motion.div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:auto-rows-[20rem]">
          {stats.map((stat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              whileHover={{ y: -8, transition: { duration: 0.2 } }}
              className={`group relative bg-white rounded-xl border border-[#f0f0f0] shadow-sm hover:shadow-xl transition-all duration-200 overflow-hidden flex flex-col ${stat.className}`}
            >
              <div className="absolute inset-0 bg-gradient-to-br from-[#e60000]/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              
              <div className="p-5 flex-1 flex flex-col">
                <div className="flex items-start justify-between mb-4">
                  <div className="p-3 rounded-lg bg-[#e60000]/10 group-hover:bg-[#e60000]/20 transition-colors">
                    <div className="text-[#e60000]">
                      {stat.icon}
                    </div>
                  </div>
                  {stat.trend && (
                    <motion.span 
                      className="text-xs font-semibold text-[#4d3c2f] bg-[#f2d88e]/30 px-2 py-1 rounded-full"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: 0.3 + index * 0.1 }}
                    >
                      {stat.trend}
                    </motion.span>
                  )}
                  {stat.percentage && (
                    <motion.span 
                      className="text-base font-bold text-[#e60000]"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: 0.3 + index * 0.1 }}
                    >
                      {stat.percentage}
                    </motion.span>
                  )}
                </div>
                
                <div className="flex-1">
                  <h3 className="text-xs font-medium text-[#7b6b59] uppercase tracking-wider mb-1">
                    {stat.title}
                  </h3>
                  
                  <div className="mb-2">
                    <p className="text-3xl font-bold text-[#1a1a1a] group-hover:text-[#e60000] transition-colors">
                      {stat.value}
                    </p>
                    {stat.subtitle && (
                      <p className="text-xs text-[#666666] mt-1">{stat.subtitle}</p>
                    )}
                  </div>
                  
                  <p className="text-xs text-[#666666] mb-3">{stat.description}</p>
                </div>
                
                {stat.details && (
                  <div className="mt-auto">
                    <div className="flex gap-3 pt-3 border-t border-[#f5f5f5]">
                      {stat.details.map((detail, idx) => (
                        <div key={idx} className="flex-1">
                          <p className="text-xs text-[#7b6b59] mb-1">{detail.label}</p>
                          <p className="text-base font-semibold text-[#1a1a1a]">{detail.value}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {stat.categories && (
                  <div className="mt-auto pt-3 border-t border-[#f5f5f5]">
                    <div className="grid grid-cols-2 gap-2">
                      {stat.categories.map((cat, idx) => (
                        <div key={idx} className="flex items-center gap-2 truncate">
                          <div 
                            className="w-2 h-2 rounded-full flex-shrink-0" 
                            style={{ backgroundColor: cat.color }}
                          />
                          <div className="flex-1 min-w-0">
                            <p className="text-xs font-medium text-[#4d3c2f] truncate">{cat.name}</p>
                            <p className="text-xs font-bold text-[#1a1a1a] truncate">{cat.value} ({cat.percentage}%)</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {stat.processMetrics && (
                  <div className="mt-auto space-y-1.5 pt-3 border-t border-[#f5f5f5]">
                    {stat.processMetrics.map((item, idx) => (
                      <div key={idx} className="flex items-center justify-between">
                        <div className="flex items-center gap-1.5 truncate">
                          <div className="relative w-2 h-2">
                            {item.active ? (
                              <motion.div 
                                className="w-2 h-2 rounded-full bg-[#e60000] absolute"
                                animate={{ scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }}
                                transition={{ duration: 2, repeat: Infinity }}
                              />
                            ) : (
                              <div className="w-2 h-2 rounded-full bg-[#e6e6e6]" />
                            )}
                          </div>
                          <span className="text-xs text-[#666666] truncate">{item.label}</span>
                        </div>
                        <span className="text-base font-bold text-[#1a1a1a]">{item.value}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default StatsDashboard;