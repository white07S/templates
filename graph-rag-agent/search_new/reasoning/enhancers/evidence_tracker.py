from typing import List, Dict, Any


class EvidenceChainTracker:
    """证据链跟踪器（占位符实现）"""
    
    def __init__(self, llm):
        """初始化证据链跟踪器"""
        self.llm = llm
        self.evidence_chain = []
    
    def add_evidence(self, evidence: str, source: str):
        """添加证据"""
        self.evidence_chain.append({"evidence": evidence, "source": source})
    
    def get_evidence_chain(self) -> List[Dict[str, Any]]:
        """获取证据链"""
        return self.evidence_chain
