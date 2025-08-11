import React, { useState } from 'react';
import { X, Send, Edit2 } from 'lucide-react';

const PromptPreview = ({ prompt, onSend, onCancel }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editedPrompt, setEditedPrompt] = useState('');
  const [userDataInput, setUserDataInput] = useState('');

  React.useEffect(() => {
    // Construct the initial prompt
    const constructedPrompt = constructPrompt(prompt, '');
    setEditedPrompt(constructedPrompt);
  }, [prompt]);

  const constructPrompt = (promptData, userData) => {
    let fullPrompt = `Persona: ${promptData.persona}\n\n`;
    fullPrompt += `Task: ${promptData.task}\n\n`;
    
    if (promptData.if_task_need_data && userData) {
      fullPrompt += `Data:\n${userData}\n\n`;
    } else if (promptData.if_task_need_data && promptData.data) {
      fullPrompt += `Data Format Example:\n${promptData.data}\n\n`;
    }
    
    fullPrompt += `Expected Response Format: ${promptData.response}`;
    
    return fullPrompt;
  };

  const handleDataInputChange = (e) => {
    const newData = e.target.value;
    setUserDataInput(newData);
    const updatedPrompt = constructPrompt(prompt, newData);
    setEditedPrompt(updatedPrompt);
  };

  const handleSend = () => {
    onSend(editedPrompt);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white w-full max-w-3xl max-h-[90vh] border-2 border-black flex flex-col">
        {/* Header */}
        <div className="p-4 border-b-2 border-black bg-gray-50">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold">Prepare Prompt</h2>
              <p className="text-sm text-gray-600 mt-1">{prompt.persona}</p>
            </div>
            <button
              onClick={onCancel}
              className="p-2 hover:bg-gray-200 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {prompt.if_task_need_data && !isEditing && (
            <div className="mb-4">
              <label className="block text-sm font-semibold mb-2">
                Provide Your Data:
              </label>
              <textarea
                value={userDataInput}
                onChange={handleDataInputChange}
                placeholder={prompt.data || "Enter your data here..."}
                rows={4}
                className="w-full p-3 border-2 border-black focus:outline-none focus:ring-2 focus:ring-gray-400"
              />
            </div>
          )}

          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-semibold">
                {isEditing ? 'Edit Prompt:' : 'Preview Prompt:'}
              </label>
              <button
                onClick={() => setIsEditing(!isEditing)}
                className="flex items-center gap-1 px-3 py-1 text-sm border-2 border-gray-300 hover:border-black transition-colors"
              >
                <Edit2 className="w-3 h-3" />
                {isEditing ? 'Preview' : 'Edit'}
              </button>
            </div>
            
            {isEditing ? (
              <textarea
                value={editedPrompt}
                onChange={(e) => setEditedPrompt(e.target.value)}
                rows={15}
                className="w-full p-3 border-2 border-black focus:outline-none focus:ring-2 focus:ring-gray-400 font-mono text-sm"
              />
            ) : (
              <div className="p-4 bg-gray-50 border-2 border-gray-300 font-mono text-sm whitespace-pre-wrap">
                {editedPrompt}
              </div>
            )}
          </div>

          {/* Keywords Display */}
          <div className="mb-4">
            <p className="text-sm font-semibold mb-2">Keywords:</p>
            <div className="flex flex-wrap gap-2">
              {prompt.keywords_used_for_search.map((keyword, index) => (
                <span
                  key={index}
                  className="px-2 py-1 text-xs border border-gray-300 bg-gray-50"
                >
                  {keyword}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t-2 border-black bg-gray-50">
          <div className="flex gap-4">
            <button
              onClick={handleSend}
              className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-black text-white hover:bg-gray-800 transition-colors font-semibold"
            >
              <Send className="w-4 h-4" />
              Send to Chat
            </button>
            <button
              onClick={onCancel}
              className="px-6 py-3 border-2 border-black hover:bg-gray-100 transition-colors font-semibold"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PromptPreview;