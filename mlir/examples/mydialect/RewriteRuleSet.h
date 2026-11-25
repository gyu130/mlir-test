#ifndef MYDIALECT_REWRITE_RULE_SET_H
#define MYDIALECT_REWRITE_RULE_SET_H

#include "mlir/IR/PatternMatch.h"
#include <functional>
#include <string>
#include <vector>

namespace mydialect {

// 单条具名图重写规则。
//   name      ：规则名，供 --mydialect-graph-rewrite="rules=a,b,c" 单独开关
//   benefit   ：模式收益，越大越优先（传给 PatternBenefit）
//   defaultOn ：未显式指定 rules 时是否默认启用
//   populate  ：把该规则对应的 pattern 注册进 RewritePatternSet
struct RewriteRule {
  std::string name;
  int benefit;
  bool defaultOn;
  std::function<void(mlir::RewritePatternSet &, mlir::MLIRContext *,
                     mlir::PatternBenefit)>
      populate;
};

// 具名规则集
struct RewriteRuleSet {
  std::string name;
  std::vector<RewriteRule> rules;
};

// 返回 mydialect 内置图重写规则集（代数化简 + 布局置换 + 算子替换规则）
RewriteRuleSet getMyDialectGraphRewriteRuleSet();

} // namespace mydialect

#endif
