from typing import Optional

import msgspec


class LogProbs(msgspec.Struct, frozen=True, tag=True):
    prompt_token_strs: list[str]
    prompt_token_logprobs: list[float | None]
    response_token_strs: list[str]
    response_token_logprobs: list[float | None]

    def __sub__(self, other: "LogProbs") -> "LogProbDiff":
        """
        Subtract another LogProbs object from this one to create a LogProbDiff.

        Args:
            other: LogProbs object to subtract from this one

        Returns:
            LogProbDiff object with computed differences (self - other)
        """
        return LogProbDiff.from_logprobs(self, other)

    def sum_response_logp(self) -> float:
        """
        Returns the mean logp of the response tokens. nans are ignored.
        """
        return sum(p for p in self.response_token_logprobs if p is not None)

    def visualize(self) -> None:
        visualize_logprobs(self)

    def trim_response_tokens(self, max_num_tokens: Optional[int] = None) -> "LogProbs":
        """
        Trim the response tokens to a maximum number of tokens.
        """
        if max_num_tokens is None:
            return self

        return LogProbs(
            prompt_token_strs=self.prompt_token_strs,
            prompt_token_logprobs=self.prompt_token_logprobs,
            response_token_strs=self.response_token_strs[:max_num_tokens],
            response_token_logprobs=self.response_token_logprobs[:max_num_tokens],
        )


class LogProbDiff(msgspec.Struct, frozen=True, tag=True):
    logp: LogProbs
    logq: LogProbs
    response_token_strs: list[str]
    response_token_logprobs: list[float | None]

    @classmethod
    def from_logprobs(cls, logp: LogProbs, logq: LogProbs) -> "LogProbDiff":
        """
        Construct a LogProbDiff from two LogProbs objects.

        Args:
            logp: First LogProbs object
            logq: Second LogProbs object (will be subtracted from logp)

        Returns:
            LogProbDiff object with computed differences
        """
        assert len(logp.response_token_strs) == len(
            logq.response_token_strs
        ), f"Token string lists have different lengths: {len(logp.response_token_strs)} vs {len(logq.response_token_strs)}"

        for i, (p_token, q_token) in enumerate(
            zip(logp.response_token_strs, logq.response_token_strs)
        ):
            assert (
                p_token == q_token
            ), f"Token strings differ at position {i}: '{p_token}' vs '{q_token}'"

        diff_logprobs: list[float | None] = []
        for p_logprob, q_logprob in zip(logp.response_token_logprobs, logq.response_token_logprobs):
            if p_logprob is None or q_logprob is None:
                diff_logprobs.append(None)
            else:
                diff_logprobs.append(p_logprob - q_logprob)

        return cls(
            logp=logp,
            logq=logq,
            response_token_strs=logp.response_token_strs,  # Since they're the same
            response_token_logprobs=diff_logprobs,
        )