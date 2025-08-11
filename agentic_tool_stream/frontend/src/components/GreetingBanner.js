import React, { useState, useEffect } from 'react';
import { Sun, Moon, Sunset, Quote } from 'lucide-react';

const GreetingBanner = ({ userName }) => {
  const [greeting, setGreeting] = useState('');
  const [icon, setIcon] = useState(null);
  const [quote, setQuote] = useState('');

  // Professional quotes suitable for banking/risk management context
  const quotes = [
    "Risk comes from not knowing what you're doing. - Warren Buffett",
    "In the business world, the rearview mirror is always clearer than the windshield. - Warren Buffett",
    "The best way to manage risk is to think about it before it happens. - Howard Marks",
    "Success is not final, failure is not fatal: it is the courage to continue that counts. - Winston Churchill",
    "Excellence is never an accident. It is always the result of high intention, sincere effort, and intelligent execution. - Aristotle",
    "The way to get started is to quit talking and begin doing. - Walt Disney",
    "Quality is not an act, it is a habit. - Aristotle",
    "Innovation distinguishes between a leader and a follower. - Steve Jobs",
    "The only way to do great work is to love what you do. - Steve Jobs",
    "Integrity is doing the right thing, even when no one is watching. - C.S. Lewis",
    "Trust takes years to build, seconds to break, and forever to repair. - Unknown",
    "Risk management is not about eliminating risk, but understanding and managing it. - Unknown",
    "Compliance is not just about following rules, it's about doing the right thing. - Unknown",
    "In banking, trust is not just important, it's everything. - Unknown",
    "The goal of risk management is not to eliminate all risks, but to make intelligent choices about which risks to take. - Unknown",
    "Prevention is better than cure, especially in risk management. - Unknown",
    "A culture of compliance starts at the top and permeates throughout. - Unknown",
    "Knowledge is power, but knowledge about risk is wisdom. - Unknown",
    "The best time to repair the roof is when the sun is shining. - John F. Kennedy",
    "By failing to prepare, you are preparing to fail. - Benjamin Franklin"
  ];

  useEffect(() => {
    const updateGreeting = () => {
      const hour = new Date().getHours();
      
      if (hour < 12) {
        setGreeting('Good morning');
        setIcon(<Sun className="w-5 h-5 text-yellow-500" />);
      } else if (hour < 17) {
        setGreeting('Good afternoon');
        setIcon(<Sunset className="w-5 h-5 text-orange-500" />);
      } else {
        setGreeting('Good evening');
        setIcon(<Moon className="w-5 h-5 text-indigo-500" />);
      }
    };

    const selectRandomQuote = () => {
      const randomIndex = Math.floor(Math.random() * quotes.length);
      setQuote(quotes[randomIndex]);
    };

    updateGreeting();
    selectRandomQuote();

    // Update greeting every minute to catch hour changes
    const interval = setInterval(updateGreeting, 60000);
    
    return () => clearInterval(interval);
  }, []);

  const displayName = userName || 'there';

  return (
    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-gray-200 p-4">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            {icon}
            <h2 className="text-xl font-semibold text-gray-800">
              {greeting}, {displayName}!
            </h2>
          </div>
          <div className="text-sm text-gray-500">
            {new Date().toLocaleDateString('en-US', { 
              weekday: 'long', 
              year: 'numeric', 
              month: 'long', 
              day: 'numeric' 
            })}
          </div>
        </div>
        <div className="flex items-start gap-2 mt-3">
          <Quote className="w-4 h-4 text-gray-400 mt-1 flex-shrink-0" />
          <p className="text-sm text-gray-600 italic">
            {quote}
          </p>
        </div>
      </div>
    </div>
  );
};

export default GreetingBanner;