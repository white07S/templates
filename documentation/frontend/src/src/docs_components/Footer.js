import React, { useEffect, useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { documentationAPI } from '../components/api';

const Footer = ({ currentPageId, onPageSelect }) => {
  const [navigation, setNavigation] = useState({
    previous: null,
    next: null,
    current: null
  });

  useEffect(() => {
    if (currentPageId) {
      fetchNavigation();
    }
  }, [currentPageId]);

  const fetchNavigation = async () => {
    try {
      const data = await documentationAPI.getNavigation(currentPageId);
      setNavigation(data);
    } catch (error) {
      console.error('Failed to fetch navigation:', error);
      setNavigation({ previous: null, next: null, current: null });
    }
  };

  return (
    <footer className="border-t border-gray-200 bg-white mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex justify-between items-center">
          {/* Previous Page */}
          <div className="flex-1">
            {navigation.previous ? (
              <button
                onClick={() => onPageSelect(navigation.previous)}
                className="group flex items-center text-gray-700 hover:text-red-600 transition-colors"
              >
                <ChevronLeft className="w-5 h-5 mr-1 group-hover:-translate-x-1 transition-transform" />
                <div className="text-left">
                  <div className="text-xs text-gray-500">Previous</div>
                  <div className="text-sm font-medium">{navigation.previous.title}</div>
                </div>
              </button>
            ) : (
              <div />
            )}
          </div>

          {/* Current Page Indicator */}
          <div className="px-4 text-center">
            {navigation.current && (
              <div className="text-xs text-gray-500">
                Current: <span className="font-medium text-gray-700">{navigation.current.title}</span>
              </div>
            )}
          </div>

          {/* Next Page */}
          <div className="flex-1 flex justify-end">
            {navigation.next ? (
              <button
                onClick={() => onPageSelect(navigation.next)}
                className="group flex items-center text-gray-700 hover:text-red-600 transition-colors"
              >
                <div className="text-right">
                  <div className="text-xs text-gray-500">Next</div>
                  <div className="text-sm font-medium">{navigation.next.title}</div>
                </div>
                <ChevronRight className="w-5 h-5 ml-1 group-hover:translate-x-1 transition-transform" />
              </button>
            ) : (
              <div />
            )}
          </div>
        </div>

        {/* Additional Footer Content */}
        <div className="mt-6 pt-6 border-t border-gray-200">
          <div className="flex justify-between items-center text-xs text-gray-500">
            <div>© 2024 Documentation System</div>
            <div className="flex space-x-4">
              <a href="#" className="hover:text-red-600 transition-colors">Privacy Policy</a>
              <a href="#" className="hover:text-red-600 transition-colors">Terms of Service</a>
              <a href="#" className="hover:text-red-600 transition-colors">Contact</a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;