#include "LLMHelper.hpp"

namespace Py = pybind11;
using namespace Py::literals;

LLMHelper::LLMHelper(llvm::StringRef apiKey, llvm::StringRef baseURL)
{
  this->mImpl = Py::module::import("llm").attr("LLMHelperImpl")(apiKey.data(),
                                                                baseURL.data());
}

std::string
LLMHelper::create_new_session()
{
  return this->mImpl.attr("create_new_session")().cast<std::string>();
}

void
LLMHelper::delete_session(llvm::StringRef sessionID)
{
  this->mImpl.attr("delete_session")(sessionID.data());
}

void
LLMHelper::add_content(llvm::StringRef sessionID,
                       Role role,
                       llvm::StringRef content)
{
  std::string _role;
  switch (role) {
    case Role::kSystem:
      _role = "system";
      break;
    case Role::kUser:
      _role = "user";
      break;
    case Role::kAssistant:
      _role = "assistant";
      break;
  }
  this->mImpl.attr("add_content")(sessionID.data(), _role, content.data());
}

std::string
LLMHelper::chat(llvm::StringRef sessionID,
                llvm::StringRef model,
                const Py::list& handlers,
                const Py::dict& params)
{
  return this->mImpl
    .attr("chat")(sessionID.data(), model.data(), handlers, **params)
    .cast<std::string>();
}
