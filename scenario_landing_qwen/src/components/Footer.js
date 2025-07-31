// src/components/Footer.js
import React from 'react';
import { motion } from 'framer-motion';

const Footer = () => {
  return (
    <footer className="bg-[#1a1a1a] text-white pt-20 pb-10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-10 mb-16">
          <div>
            <div className="flex items-center mb-6">
              <div className="bg-[#d6001c] p-2 rounded-lg mr-3">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className="h-6 w-6 fill-white">
                  <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path>
                </svg>
              </div>
              <h3 className="text-xl font-bold">GCRG Scenario</h3>
            </div>
            <p className="text-[#b3b3b3] mb-6">
              Advanced AI-powered risk scenario generation and management platform for financial institutions.
            </p>
            <div className="flex space-x-4">
              {['LinkedIn', 'Twitter', 'GitHub'].map((social, index) => (
                <a 
                  key={index} 
                  href="#" 
                  className="bg-[#333] hover:bg-[#d6001c] w-10 h-10 rounded-full flex items-center justify-center transition-colors"
                >
                  {social.charAt(0)}
                </a>
              ))}
            </div>
          </div>
          
          <div>
            <h4 className="text-lg font-bold mb-6">Support & Contact</h4>
            <ul className="space-y-4">
              <li>
                <a href="#" className="text-[#b3b3b3] hover:text-white transition-colors">Access Management</a>
              </li>
              <li>
                <a href="#" className="text-[#b3b3b3] hover:text-white transition-colors">Bug Reporting</a>
              </li>
              <li>
                <a href="#" className="text-[#b3b3b3] hover:text-white transition-colors">Design Changes</a>
              </li>
              <li>
                <a href="#" className="text-[#b3b3b3] hover:text-white transition-colors">Feature Requests</a>
              </li>
            </ul>
          </div>
          
          <div>
            <h4 className="text-lg font-bold mb-6">Resources</h4>
            <ul className="space-y-4">
              <li>
                <a href="#" className="text-[#b3b3b3] hover:text-white transition-colors">Documentation</a>
              </li>
              <li>
                <a href="#" className="text-[#b3b3b3] hover:text-white transition-colors">API Reference</a>
              </li>
              <li>
                <a href="#" className="text-[#b3b3b3] hover:text-white transition-colors">User Guides</a>
              </li>
              <li>
                <a href="#" className="text-[#b3b3b3] hover:text-white transition-colors">Release Notes</a>
              </li>
            </ul>
          </div>
          
          <div>
            <h4 className="text-lg font-bold mb-6">Contact Us</h4>
            <address className="not-italic text-[#b3b3b3] space-y-4">
              <div>
                <div className="font-medium text-white">Primary Contact:</div>
                <a href="mailto:preetam.sharma@ubs.com" className="hover:text-white transition-colors">
                  preetam.sharma@ubs.com
                </a>
              </div>
              <div>
                <div className="font-medium text-white">Governance Team:</div>
                <a href="mailto:gcrg.governance@ubs.com" className="hover:text-white transition-colors">
                  gcrg.governance@ubs.com
                </a>
              </div>
              <div>
                <div className="font-medium text-white">Support Hotline:</div>
                <div>+41 44 234 11 22</div>
              </div>
            </address>
          </div>
        </div>
        
        <div className="border-t border-[#333] pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="text-[#b3b3b3] text-sm mb-4 md:mb-0">
              © 2023 UBS Group AG. All rights reserved.
            </div>
            <div className="flex space-x-6">
              <a href="#" className="text-[#b3b3b3] hover:text-white text-sm transition-colors">Privacy Policy</a>
              <a href="#" className="text-[#b3b3b3] hover:text-white text-sm transition-colors">Terms of Service</a>
              <a href="#" className="text-[#b3b3b3] hover:text-white text-sm transition-colors">Compliance</a>
              <a href="#" className="text-[#b3b3b3] hover:text-white text-sm transition-colors">Cookies</a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;