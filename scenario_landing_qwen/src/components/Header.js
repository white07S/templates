// src/components/Header.js
import React, { useState, useEffect } from 'react';
import { Home, Library, Wand, X, Menu } from 'lucide-react';

const Header = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navItems = [
    { name: "Home", link: "#", icon: <Home className="h-4 w-4" /> },
    { name: "Scenario Library", link: "#library", icon: <Library className="h-4 w-4" /> },
    { name: "Scenario Generator", link: "#generator", icon: <Wand className="h-4 w-4" /> }
  ];

  return (
    <header className="sticky top-0 z-50">
      <div className={`transition-all duration-300 ${isScrolled ? 'py-2 bg-white shadow-md' : 'py-4 bg-transparent'}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <div className="bg-[#d6001c] p-2 rounded-lg mr-3">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className="h-6 w-6 fill-white">
                  <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path>
                </svg>
              </div>
              <h1 className="text-2xl font-bold text-[#d6001c]">GCRG Scenario</h1>
            </div>
            
            {/* Desktop Navigation */}
            <nav className="hidden md:flex space-x-8">
              {navItems.map((item, index) => (
                <a 
                  key={index} 
                  href={item.link}
                  className="flex items-center text-[#333] hover:text-[#d6001c] transition-colors"
                >
                  <span className="mr-2">{item.icon}</span>
                  {item.name}
                </a>
              ))}
            </nav>
            
            <div className="hidden md:block">
              <button className="bg-[#d6001c] hover:bg-[#a80016] text-white font-medium py-2 px-4 rounded-lg transition-colors">
                Login
              </button>
            </div>
            
            {/* Mobile menu button */}
            <button 
              className="md:hidden text-[#333]"
              onClick={() => setMobileMenuOpen(true)}
            >
              <Menu className="h-6 w-6" />
            </button>
          </div>
        </div>
      </div>
      
      {/* Mobile menu */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50">
          <div className="absolute right-0 top-0 bottom-0 w-64 bg-white p-4">
            <div className="flex justify-end mb-6">
              <button onClick={() => setMobileMenuOpen(false)}>
                <X className="h-6 w-6 text-[#333]" />
              </button>
            </div>
            <div className="flex flex-col space-y-6">
              {navItems.map((item, index) => (
                <a 
                  key={index} 
                  href={item.link}
                  className="flex items-center text-[#333] hover:text-[#d6001c] transition-colors"
                  onClick={() => setMobileMenuOpen(false)}
                >
                  <span className="mr-2">{item.icon}</span>
                  {item.name}
                </a>
              ))}
              <button className="bg-[#d6001c] hover:bg-[#a80016] text-white font-medium py-2 px-4 rounded-lg transition-colors mt-4">
                Login
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
};

export default Header;